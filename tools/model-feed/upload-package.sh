#!/usr/bin/env bash
# Uploads the objects of one published package version from a local feed folder to the R2 bucket (docs/model-feed.md, "Publishing the
# official feed"); used by .github/workflows/model-feed-publish.yml.
#
#   upload-package.sh <work dir> <staging folder> <id> <version>
#
# Order: the package file first, then manifest.json and manifest.sig, so a manifest is never visible before its file. Idempotent and
# safe to rerun after an interruption: an object that exists with the same content is kept (an earlier, interrupted run stored it; a
# multipart upload only becomes an object when complete; package files carry their SHA-256 as object metadata `sha256`, documents are
# compared byte for byte), one with other content fails the run (published objects are never replaced).
# A manifest-only version (reuse.json) copies the earlier version's package object server-side instead of uploading it.
#
# Needs: aws (R2_BUCKET, R2_ENDPOINT, FEED_PREFIX and the AWS_* credentials in the environment), jq.
set -euo pipefail

work=$1
folder=$2
id=$3
version=$4
version_dir="packages/$id/$version"
immutable="public, max-age=31536000, immutable"

remote_size() { # key -> size, or empty when the object does not exist
  aws s3api head-object --bucket "$R2_BUCKET" --key "$FEED_PREFIX/$1" --endpoint-url "$R2_ENDPOINT" --query ContentLength --output text 2>/dev/null || true
}

content_type() {
  case "$1" in
    *.json) echo application/json ;;
    *.sig) echo application/jose ;;
    *) echo application/octet-stream ;;
  esac
}

remote_sha256() { # key -> the sha256 metadata of the object, or empty
  aws s3api head-object --bucket "$R2_BUCKET" --key "$FEED_PREFIX/$1" --endpoint-url "$R2_ENDPOINT" --query Metadata.sha256 --output text 2>/dev/null || true
}

same_content() { # key, local file: the stored object is this file (package files by their sha256 metadata, documents byte for byte)
  local key=$1 file=$2
  case "$key" in
    *.slm)
      [ "$(remote_sha256 "$key")" = "$(jq -r '.files[] | select(.role == null or .role == "package") | .sha256' "$work/feed/$version_dir/manifest.json" | head -1)" ] ;;
    *)
      local copy
      copy=$(mktemp)
      aws s3 cp "s3://$R2_BUCKET/$FEED_PREFIX/$key" "$copy" --endpoint-url "$R2_ENDPOINT" --only-show-errors
      cmp -s "$copy" "$file"
      local same=$?
      rm -f "$copy"
      return $same ;;
  esac
}

put() { # relative path in the feed folder
  local key=$1 local_size existing metadata=()
  local_size=$(stat -c %s "$work/feed/$key")
  existing=$(remote_size "$key")
  if [ -n "$existing" ] && [ "$existing" != "None" ]; then
    if [ "$existing" = "$local_size" ] && same_content "$key" "$work/feed/$key"; then
      echo "$key: stored already (same content), kept."
      return 0
    fi
    echo "::error::$key exists with other content ($existing bytes, local $local_size); published objects are never replaced. Build a new version."
    return 1
  fi
  case "$key" in
    *.slm) metadata=(--metadata "sha256=$(jq -r '.files[] | select(.role == null or .role == "package") | .sha256' "$work/feed/$version_dir/manifest.json" | head -1)") ;;
  esac
  # A whole-object retry on top of the CLI's per-request retries: one failed part of a 100+ GB multipart upload (a throughput or
  # connection error) fails the copy, and the object only exists once the upload completes, so starting it again is safe.
  local attempt
  for attempt in 1 2 3 4; do
    if aws s3 cp "$work/feed/$key" "s3://$R2_BUCKET/$FEED_PREFIX/$key" --endpoint-url "$R2_ENDPOINT" --only-show-errors \
      --content-type "$(content_type "$key")" --cache-control "$immutable" "${metadata[@]}"; then
      echo "$key: uploaded ($local_size bytes)."
      return 0
    fi
    echo "::warning::$key: upload attempt $attempt failed."
    [ "$attempt" -lt 4 ] && sleep $((attempt * 60))
  done
  echo "::error::$key: upload failed after 4 attempts."
  return 1
}

package_file=$(jq -r '.files[] | select(.role == null or .role == "package") | .name' "$work/feed/$version_dir/manifest.json" | head -1)
package_size=$(jq -r '.files[] | select(.role == null or .role == "package") | .size' "$work/feed/$version_dir/manifest.json" | head -1)

# 1. The package file.
if [ -f "$folder/reuse.json" ]; then
  from_version=$(jq -r .version "$folder/reuse.json")
  from_file=$(jq -r .file "$folder/reuse.json")
  existing=$(remote_size "$version_dir/$package_file")
  if [ -z "$existing" ] || [ "$existing" = "None" ]; then
    aws s3 cp "s3://$R2_BUCKET/$FEED_PREFIX/packages/$id/$from_version/$from_file" "s3://$R2_BUCKET/$FEED_PREFIX/$version_dir/$package_file" \
      --endpoint-url "$R2_ENDPOINT" --only-show-errors --metadata-directive REPLACE --content-type application/octet-stream --cache-control "$immutable" \
      --metadata "sha256=$(jq -r '.files[] | select(.role == null or .role == "package") | .sha256' "$work/feed/$version_dir/manifest.json" | head -1)"
    echo "$version_dir/$package_file: copied from $from_version."
  fi
elif [ -f "$work/feed/$version_dir/$package_file" ]; then
  put "$version_dir/$package_file"
else
  # Only the stored manifest was staged (the feed has this exact package): the object must exist already.
  existing=$(remote_size "$version_dir/$package_file")
  if [ "$existing" != "$package_size" ]; then
    echo "::error::$version_dir/$package_file is not in the bucket (or has another size) and is not available locally; rebuild it."
    exit 1
  fi
fi

# 2. The manifest and its signature.
put "$version_dir/manifest.json"
put "$version_dir/manifest.sig"

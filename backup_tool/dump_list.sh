#!/bin/bash
source "$(dirname "$0")/.env"
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY

echo "📦 최근 백업 목록 (최대 10개):"
aws s3 ls s3://$S3_BUCKET/dump/ | sort -rk1,2 | head -n 10

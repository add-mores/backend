#!/bin/bash
NOW=$(date +"%Y%m%d_%H%M")
DUMP_FILE="${NOW}_server.dump"

# 환경 변수 설정
source "$(dirname "$0")/.env"
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export PGPASSWORD=$RDS_PASSWORD


# 덤프 생성
pg_dump -h $RDS_HOST -U $RDS_USER -d $RDS_DB -F c -b -f $DUMP_FILE

# 업로드
aws s3 cp $DUMP_FILE s3://$S3_BUCKET/dump/

# 정리
rm $DUMP_FILE

echo "✅ 서버 백업 완료: $DUMP_FILE"


#!/bin/bash
USER_NAME=$1
if [ -z "$USER_NAME" ]; then
  echo "❗ 사용자명을 인수로 입력하세요."
  exit 1
fi

DB_USER="devuser"
NOW=$(date +"%Y%m%d_%H%M")
DUMP_FILE="${NOW}_${USER_NAME}.dump"
CONTAINER_NAME="dev_db"
DB_NAME="dev_db"

source "$(dirname "$0")/.env"
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY

# 덤프 생성
docker exec $CONTAINER_NAME pg_dump -U $DB_USER -d $DB_NAME -Fc -b -f /tmp/$DUMP_FILE

# 컨테이너에서 복사
docker cp $CONTAINER_NAME:/tmp/$DUMP_FILE ./$DUMP_FILE

# 업로드
aws s3 cp ./$DUMP_FILE s3://$S3_BUCKET/dump/

# 정리
rm $DUMP_FILE

echo "✅ 로컬 백업 완료: $DUMP_FILE"

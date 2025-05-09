#!/bin/bash
DUMP_FILE=$1
MODE=$2

if [ -z "$DUMP_FILE" ]; then
  echo "❗ 복원할 dump 파일명을 입력하세요."
  exit 1
fi

source "$(dirname "$0")/.env"
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY

CONTAINER="dev_db"
TMP_PATH="/tmp/$DUMP_FILE"

# S3에서 다운로드
aws s3 cp s3://$S3_BUCKET/dump/$DUMP_FILE $DUMP_FILE

# 컨테이너에 복사
docker cp $DUMP_FILE $CONTAINER:$TMP_PATH

# -d: 전체 삭제 후 복원, -c: 기존 테이블 유지/병합
if [ "$MODE" == "-d" ]; then
  docker exec $CONTAINER dropdb -U devuser dev_db
  docker exec $CONTAINER createdb -U devuser dev_db
fi

docker exec $CONTAINER pg_restore -U devuser -d dev_db $TMP_PATH

# 정리
rm $DUMP_FILE
docker exec $CONTAINER rm $TMP_PATH

echo "✅ 로컬 복원 완료"


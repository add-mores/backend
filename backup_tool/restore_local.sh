#!/bin/bash
DUMP_FILE=$1
MODE=$2
TARGET_TABLE=$3

if [ -z "$DUMP_FILE" ]; then
  echo "❗ 복원할 dump 파일명을 입력하세요."
  exit 1
fi

source "$(dirname "$0")/.env"
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY

CONTAINER="dev_db"
TMP_PATH="/tmp/$DUMP_FILE"

DB_NAME="dev_db"
DB_USER="devuser"

# S3에서 다운로드
aws s3 cp s3://$S3_BUCKET/dump/$DUMP_FILE $DUMP_FILE

# 컨테이너에 복사
docker cp $DUMP_FILE $CONTAINER:$TMP_PATH

# -d: 전체 삭제 후 복원, -c: 기존 테이블 유지/병합
if [ "$MODE" == "-d" ]; then
  docker exec $CONTAINER dropdb -U devuser dev_db
  docker exec $CONTAINER createdb -U devuser dev_db
  docker exec $CONTAINER pg_restore -U $DB_USER -d $DB_NAME $TMP_PATH
elif [ "$MODE" == "-c" ]; then
  if [ -z "$TARGET_TABLE" ]; then
    echo "❗ -c 옵션에는 대상 테이블명을 추가로 입력해야 합니다."
    exit 1
  fi
  # 테이블이 존재하는지 확인 후 있으면 삭제
  docker exec $CONTAINER psql -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS \"$TARGET_TABLE\" CASCADE;"

  # 덤프에서 해당 테이블만 복원
  docker exec $CONTAINER pg_restore -U $DB_USER -d $DB_NAME -t "$TARGET_TABLE" $TMP_PATH
else
  echo "❗ 올바른 옵션을 입력하세요. -d 또는 -c <테이블명>"
  exit 1
fi

# 정리
rm $DUMP_FILE
docker exec $CONTAINER rm $TMP_PATH

echo "✅ 로컬 복원 완료"


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
export PGPASSWORD=$RDS_PASSWORD

# 서버 백업 먼저 수행
BACKUP_NAME=$(date +"%Y%m%d_%H%M")_server_autobackup.dump
pg_dump -h $RDS_HOST -U $RDS_USER -d $RDS_DB -F c -b -f $BACKUP_NAME
aws s3 cp $BACKUP_NAME s3://$S3_BUCKET/dump/
rm $BACKUP_NAME

# 대상 덤프 다운로드
aws s3 cp s3://$S3_BUCKET/dump/$DUMP_FILE $DUMP_FILE

# 복원 모드 분기
if [ "$MODE" == "-d" ]; then
  dropdb -h $RDS_HOST -U $RDS_USER $RDS_DB
  createdb -h $RDS_HOST -U $RDS_USER $RDS_DB
fi

pg_restore -h $RDS_HOST -U $RDS_USER -d $RDS_DB $DUMP_FILE

# 정리
rm $DUMP_FILE

echo "✅ 서버 복원 완료"


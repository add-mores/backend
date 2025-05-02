import subprocess
import requests
import toml
import sys

WEBHOOK_URL = "https://discordapp.com/api/webhooks/1367855332998643852/xGx8hG9rggJbUVBRod_8BgI69Rg1lC0yqktz4JNG4qyNcOw5t65IdaSfPPEOd_FNpCBZ"

# 1. git add 된 파일 목록 가져오기
def get_added_files():
    result = subprocess.run(
        ["git", "diff", "--name-only", "--cached"],
        capture_output=True,
        text=True
    )

    if not result.stdout.strip():
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True
        )
        files = [
            line.split()[-1] for line in result.stdout.strip().splitlines() if line.startswith("A") or line.startswith("M")
        ]
    else:
        files = result.stdout.strip().splitlines()
    return files




# 2. push되지 않은 커밋 로그 (현재 브랜치 기준)
def get_commits():
    try:
        result = subprocess.run(
            ["git", "log", "@{u}..HEAD", "--pretty=format:- %s%n%b", "--reverse"],
            capture_output=True,
            text=True,
            check=True
        )
        #formatted_commits = result.stdout.strip().replace("\n", "\\n")
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # upstream이 없는 경우
        print("❌ 업스트림 브랜치가 설정되어 있지 않아 push되지 않은 커밋을 확인할 수 없습니다.")
        print("💡 `git push -u origin <branch>`로 upstream을 먼저 설정해주세요.")
        sys.exit(1)
        

# 3. pyproject.toml에서 버전 가져오기
def get_version_from_pyproject():
    try:
        with open("pyproject.toml", "r") as f:
            data = toml.load(f)
            return data.get("project", {}).get("version", "unknown")
    except FileNotFoundError:
        return "pyproject.toml not found"

# 4. 디스코드 전송 함수
def send_to_discord(files, commits, version):
    commit_log = commits if commits else "없음"
    content = f"""
---------------------------------------------------------------------
🚀*패치노트 자동 푸시 알림*

**버전**: `{version}`

**커밋 내역**:
```
{commit_log}
```
"""
    requests.post(WEBHOOK_URL, json={"content": content})

# 5. 실행
if __name__ == "__main__":
    added_files = get_added_files()
    commits = get_commits()
    version = get_version_from_pyproject()
    send_to_discord(added_files, commits, version)

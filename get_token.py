import subprocess
try:
    p = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n", capture_output=True, text=True, check=True)
    out = p.stdout
    for line in out.split('\n'):
        if line.startswith('password='):
            token = line.split('=', 1)[1].strip()
            print(f"TOKEN_FOUND:{token[:5]}...")
            with open('token.txt', 'w') as f:
                f.write(token)
            break
except Exception as e:
    print(e)

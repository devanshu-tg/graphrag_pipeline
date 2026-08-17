#!/usr/bin/env bash
# One-shot setup for an Oracle Cloud Always Free ARM VM (Ubuntu 22.04/24.04).
#
#     sudo bash bootstrap.sh
#
# Installs Docker and Caddy, opens the firewall properly (see the iptables note below),
# builds the image for ARM64, and installs a systemd unit so the service survives a reboot.
#
# It does NOT create secrets. Fill /opt/graphrag/secrets.env yourself before the final step;
# the script stops and tells you if it is missing or still has placeholders in it.
set -euo pipefail

APP_DIR=/opt/graphrag
SRC_DIR="${APP_DIR}/src"
SECRETS="${APP_DIR}/secrets.env"

log() { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

log "Docker"
if ! command -v docker >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg git git-lfs
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker
fi
docker --version

# THE ORACLE GOTCHA. Opening 80/443 in the VCN security list is only half of it: Oracle's
# Ubuntu images ship an iptables INPUT chain that REJECTs everything except SSH, and it
# survives a reboot via netfilter-persistent. Every "I opened the port and it still times
# out" report about Oracle is this. Both layers have to be opened, and the VCN half is done
# in the web console — this script cannot do it for you.
log "firewall (host side)"
if command -v iptables >/dev/null; then
    for port in 80 443; do
        if ! iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
            # insert BEFORE the catch-all REJECT rather than appending after it
            iptables -I INPUT 6 -p tcp --dport "$port" -m state --state NEW,ESTABLISHED -j ACCEPT
            echo "  opened $port"
        else
            echo "  $port already open"
        fi
    done
    command -v netfilter-persistent >/dev/null && netfilter-persistent save || \
        { apt-get install -y -qq iptables-persistent && netfilter-persistent save; }
fi
echo "  REMINDER: also add ingress for 80 and 443 to the VCN security list in the Oracle console."

log "Caddy"
if ! command -v caddy >/dev/null; then
    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    apt-get update -qq && apt-get install -y -qq caddy
fi

log "image"
[ -d "$SRC_DIR" ] || { echo "ERROR: ${SRC_DIR} missing — upload the project there first"; exit 1; }
for f in datasets/enterprise/corpus.jsonl pipeline3_round3/index_v2/edges_prov.json; do
    [ -s "${SRC_DIR}/enterprise-graphrag/${f}" ] || {
        echo "ERROR: missing ${f} — it is gitignored for size, copy it across explicitly"; exit 1; }
done
cd "${SRC_DIR}/enterprise-graphrag"
# The Dockerfile picks stock PyPI torch on aarch64 and the CPU index on x86_64, so the same
# file builds on both. Expect 15-25 min on 2 free-tier ARM cores.
docker build -f deploy/Dockerfile -t graphrag-demo .

log "secrets"
if [ ! -f "$SECRETS" ]; then
    cat > "$SECRETS" <<'EOF'
GOOGLE_API_KEY=REPLACE
TIGERGRAPH_HOST=REPLACE
TIGERGRAPH_SECRET=REPLACE
TIGERGRAPH_USERNAME=tigergraph
TIGERGRAPH_PASSWORD=REPLACE
DEMO_TOKEN=REPLACE
EOF
    chmod 600 "$SECRETS"
    echo "  wrote ${SECRETS} — fill it in, then re-run this script"
    exit 0
fi
if grep -q REPLACE "$SECRETS"; then
    echo "ERROR: ${SECRETS} still contains REPLACE placeholders"; exit 1
fi
chmod 600 "$SECRETS"

log "systemd unit"
cat > /etc/systemd/system/graphrag.service <<EOF
[Unit]
Description=GraphRAG live demo
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=10
ExecStartPre=-/usr/bin/docker rm -f graphrag
# Bound to 127.0.0.1: the ONLY way in is Caddy, which terminates TLS and applies CORS.
# Publishing 0.0.0.0:8090 would expose the unencrypted service directly and let anyone
# bypass both.
ExecStart=/usr/bin/docker run --rm --name graphrag \\
    -p 127.0.0.1:8090:8000 \\
    --env-file ${SECRETS} \\
    -e DEMO_RATE_PER_HOUR=20 \\
    --memory=3g \\
    graphrag-demo
ExecStop=/usr/bin/docker stop graphrag

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now graphrag

log "done"
IP=$(curl -s --max-time 10 ifconfig.me || echo "<your-ip>")
echo "  service   : systemctl status graphrag"
echo "  logs      : docker logs -f graphrag"
echo "  local test: curl localhost:8090/api/status     (expect {\"state\":\"ready\"} in ~60s)"
echo
echo "  NEXT: put your IP into the Caddyfile with dashes and start Caddy:"
echo "     sed -i 's/REPLACE-WITH-DASHED-IP/${IP//./-}/' /etc/caddy/Caddyfile"
echo "     systemctl reload caddy"
echo "  then: https://graphrag.${IP//./-}.sslip.io/api/status"

#!/usr/bin/env bash
set -euo pipefail

# Docker install script for Ubuntu 24.04 (noble)

echo "[1/6] Remove old Docker packages (if any)"
sudo apt-get remove -y docker docker-engine docker.io containerd runc || true

echo "[2/6] Install dependencies"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release

echo "[3/6] Add Docker GPG key"
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "[4/6] Add Docker apt repo (noble)"
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu noble stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

echo "[5/6] Install Docker Engine + plugins"
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[6/6] Enable Docker, add current user to docker group, and verify"
sudo systemctl enable --now docker

# Allow running docker without sudo (requires re-login to fully apply)
sudo usermod -aG docker "$USER"

echo
echo "Docker version:"
docker --version || true

echo
echo "Running hello-world (may require sudo until you re-login):"
if docker run --rm hello-world >/dev/null 2>&1; then
  docker run --rm hello-world
else
  sudo docker run --rm hello-world
fi

echo
echo "Done. If 'docker run' required sudo, log out/in (or reboot) for group change to take effect."
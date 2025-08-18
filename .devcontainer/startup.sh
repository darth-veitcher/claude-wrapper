#!/bin/bash
# This script is executed when the devcontainer starts.
set -e

echo "Starting devcontainer setup..."
echo "Current directory: $(pwd)"
echo "Current user: $(whoami)"
echo "Container hostname: $(hostname)"
echo "------------------------------"

cd /workspace

# Ensure the workspace is clean and up-to-date
echo "Updating git submodules..."
git submodule update --init --recursive
git submodule foreach --recursive git fetch
mkdir -p .devcontainer/.data

# Build mcp-memgraph image
echo "Building mcp-memgraph Docker image..."
docker build -f .devcontainer/mcp/memgraph-ai-toolkit/integrations/mcp-memgraph/Dockerfile \
    -t mcp-memgraph:latest \
    .devcontainer/mcp/memgraph-ai-toolkit

# Start MCP services on isolated network
echo "Starting MCP services..."
docker compose -f .devcontainer/compose-mcp.yml down --remove-orphans || true
sudo chown -R $(id -u):$(id -g) .devcontainer/.data
docker compose -f .devcontainer/compose-mcp.yml up -d

# Wait for services to start
echo "Waiting for services to initialize..."
sleep 15

# Map container IPs to /etc/hosts for hostname resolution
echo "=== Mapping Service IPs for Hostname Resolution ==="

map_service() {
    local container_name=$1
    local hostname=$2

    local ip=$(docker inspect $container_name 2>/dev/null | jq -r '.[0].NetworkSettings.Networks["mcp-services"].IPAddress' 2>/dev/null)

    if [ "$ip" != "null" ] && [ -n "$ip" ]; then
        # Remove existing entries
        sudo sed -i "/$hostname/d" /etc/hosts 2>/dev/null || true

        # Add new entry
        echo "$ip $hostname" | sudo tee -a /etc/hosts > /dev/null
        echo "✓ $hostname -> $ip"
        return 0
    else
        echo "✗ Could not map $container_name"
        return 1
    fi
}

# Map all services
map_service "mcp-redis" "redis"
map_service "mcp-crawl4ai" "crawl4ai"
map_service "mcp-searxng" "searxng"
map_service "mcp-context7" "context7-mcp"
# mcp-memgraph is run on-demand via docker command, not mapped
map_service "memgraph" "memgraph"
map_service "memgraph-lab" "memgraph-lab"

# Test connectivity
echo ""
echo "=== Testing Service Connectivity ==="

for service_port in "redis:6379" "crawl4ai:11235" "searxng:8080" "context7-mcp:8080" "memgraph:7687" "memgraph-lab:3000"; do
    hostname=$(echo $service_port | cut -d: -f1)
    port=$(echo $service_port | cut -d: -f2)

    echo -n "Testing $hostname:$port... "
    if nc -z $hostname $port 2>/dev/null; then
        echo "✓"
    else
        echo "✗"
    fi
done

# Quick HTTP health checks
echo ""
echo "=== HTTP Health Checks ==="
curl -f http://crawl4ai:11235/health >/dev/null 2>&1 && echo "✓ Crawl4AI healthy" || echo "✗ Crawl4AI not responding"
wget --quiet --spider http://searxng:8080/ >/dev/null 2>&1 && echo "✓ SearXNG healthy" || echo "✗ SearXNG not responding"

# Update tools
echo "Updating tools..."
claude update
uvx superclaude update --quiet --yes --components commands core mcp || uvx SuperClaude install --profile developer --yes --quiet || true

echo ""
echo "------------------------------"
echo "✅ DevContainer setup completed!"
echo ""
echo "🔗 Services accessible by hostname (via /etc/hosts mapping):"
echo "  • Crawl4AI:     http://crawl4ai:11235"
echo "  • SearXNG:      http://searxng:8080"
echo "  • Redis:        redis:6379"
echo "  • Context7:     http://context7-mcp:8080"
echo "  • Memgraph:     bolt://memgraph:7687"
echo "  • Memgraph Lab: http://memgraph-lab:3000"
echo ""
echo "🌐 Host access (if needed):"
echo "  • Crawl4AI:     http://localhost:11235"
echo "  • SearXNG:      http://localhost:8080"
echo "  • Memgraph Lab: http://localhost:3000"
echo "------------------------------"

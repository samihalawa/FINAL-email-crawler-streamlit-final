#!/bin/bash
GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"
BASE_URL="http://localhost:8000"

test_endpoint() {
    echo -e "\n${GREEN}Testing: $1${NC}"
    eval "$2"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test passed${NC}"
    else
        echo -e "${RED}✗ Test failed${NC}"
    fi
}
test_endpoint "Bulk search" "curl -s -X POST $BASE_URL/bulk-search -H \"Content-Type: application/json\" -d \"{\\"terms\\":[\\"software development\\"],\\"num_results\\":2,\\"optimize_english\\":true}\"" || true
test_endpoint "Search term optimization" "curl -s -X POST $BASE_URL/search-terms/optimize -H \"Content-Type: application/json\" -d \"{\\"terms\\":[\\"software development\\"]}\"" || true
test_endpoint "Create email settings" "curl -s -X POST $BASE_URL/settings/email -H \"Content-Type: application/json\" -d \"{\\"name\\":\\"Test SMTP\\",\\"email\\":\\"test@example.com\\",\\"provider\\":\\"smtp\\",\\"smtp_server\\":\\"smtp.example.com\\",\\"smtp_port\\":587}\"" || true
test_endpoint "Set active project" "curl -s -X POST $BASE_URL/set-active-project -H \"Content-Type: application/json\" -d \"{\\"project_id\\":1}\"" || true
test_endpoint "Set active campaign" "curl -s -X POST $BASE_URL/set-active-campaign -H \"Content-Type: application/json\" -d \"{\\"campaign_id\\":1}\"" || true

name: CRT Discord Alert

# 15분마다 자동 실행
on:
  schedule:
    - cron: "*/15 * * * *"
  workflow_dispatch: {}

jobs:
  check-signal:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r .github/workflows/requirements.txt

      - name: Restore alert-state cache
        uses: actions/cache@v4
        with:
          path: last_alert_state.json
          key: crt-alert-state

      - name: Run signal check
        env:
          TWELVEDATA_API_KEY: ${{ secrets.TWELVEDATA_API_KEY }}
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
          INSTRUMENT: XAU/USD
          ACCOUNT_EQUITY: "10000"
          RISK_PERCENT: "0.5"
        run: python .github/workflows/crt_discord_alert.py --once

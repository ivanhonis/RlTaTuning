echo --------------------------------------
echo Kill all running Python process
echo --------------------------------------

ps aux
for KILLPID in `ps ax | grep 'python3' | awk ' { print $1;}'`; do
  echo kill pid $KILLPID;
  kill -9 $KILLPID;
done

read -p "Do you want to run TRADE_SERVER_FUTURES.py in the background? (y/n): " choice

if [ "$choice" = "y" ]; then
  echo --------------------------------------
  echo RUN RL_trades_server.py IN BACKGROUND
  echo --------------------------------------
  nohup python3 -u RL_trades_server.py > output.log 2>&1 &
else
  echo --------------------------------------
  echo RUN FCS
  echo --------------------------------------
  python3 RL_trades_server.py
fi


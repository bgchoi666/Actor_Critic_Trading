import pandas as pd
import numpy as np

import datetime
import sys

result_path = 'ensemble/pred_83_results.csv'

pred_term = 5

loss_cut = float(0.0003)
trading_9h = False

slippage = 0.05
margin = 10000000

def calc_profit():
    df = pd.read_csv(result_path, encoding='euc-kr')

    # 15시 종가를 익일 시가로 조정
    #for i in range(len(df)-1):
    #    if i != 0 and df.loc[i, 'date'][11:13] == '15':
    #        df.loc[i, 'close'] = df.loc[i+1, 'open']

    state = 0
    count = 0
    buy_price = 0
    profit = []
    fee = []

    EOF = False

    for i in range(len(df)):

        pred = int(df.loc[i, 'result'])
        close = float(df.loc[i, 'close'])
        high = float(df.loc[i, 'high'])
        low = float(df.loc[i, 'low'])
        open = float(df.loc[i, 'open'])
        date = df.loc[i, 'date']

        if trading_9h:
            if date[11:13] == '09' and state != 0:
                buy_price = open

        if date[11:13] == '15':
            if state == 1:
                profit.append((buy_price - close - slippage) * 250000)
                fee.append((buy_price + close) * 250000 * 0.00003)
            elif state == 2:
                profit.append((close - buy_price - slippage) * 250000)
                fee.append((buy_price + close) * 250000 * 0.00003)
            else:
                profit.append(0)
                fee.append(0)
            if trading_9h:
                state = pred
                count = 1
                buy_price = close
            else:
                state = 0
                count = 0
                buy_price = 0
            EOF = False
            continue

        if EOF:
            profit.append(0)
            fee.append(0)
            continue

        if state == 1:

            if high - buy_price >= buy_price * loss_cut:
                profit.append(-int((buy_price*loss_cut+slippage)/0.05)*0.05*250000)
                fee.append((buy_price + close)*250000*0.00003)
                EOF = True
                state = 0
                #state = pred
                #count = 1
                #buy_price = close
            elif pred == 2:
                profit.append((buy_price - close - slippage)*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = pred
                count = 1
                buy_price = close
            else:
                profit.append(0)
                fee.append(0)
                count += 1
                count %= pred_term
        elif state == 2:

            if buy_price - low >= buy_price * loss_cut:
                profit.append(-int((buy_price*loss_cut+slippage)/0.05)*0.05*250000)
                fee.append((buy_price + close)*250000*0.00003)
                EOF = True
                state = 0
                #state = pred
                #count = 1
                #buy_price = close
            elif pred == 1:
                profit.append((close - buy_price - slippage)*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = pred
                count = 1
                buy_price = close
            else:
                profit.append(0)
                fee.append(0)
                count += 1
                count %= pred_term
        else:
            if pred == 1 or pred == 2:
                state = pred
                count = 1
                buy_price = close
            else:
                count = 0
            profit.append(0)
            fee.append(0)

        #if (sum(profit) - sum(fee)) / close / 250000 / 0.09 + 1 < 0.8:
        #    df = df[:i+1]
        #    break

    df['profit'] = profit
    df['fee'] = fee

    df.to_csv(result_path, index=False, encoding='euc-kr')

    return (sum(profit) - sum(fee))/margin+1
if __name__ == "__main__":
    if len(sys.argv) > 1:
        pred_term = int(sys.argv[1])
    r = calc_profit()
    print(r)
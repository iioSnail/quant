"""
想象一下，现在有两个LOL战队正在比赛, 人头比目前是0-1。
两队拿五杀（优先第五个人头）的赔率分别为2.7和1.5。
假设两队每拿一个人头的概率都是50%，那么我是否应该下注。

结论：
1. 当比分0-1时，若赔率>2.7，且0分方实力或阵容较好（拿人头的概率>50%），则可以获利。
2. 不要进行双向压，虽然可以保证那一局稳赚不赔，但总体来说赢得会少。

Imagine that two teams in a League of Legends
match are currently competing, with the score
being 0-1 in kills. The odds of the two teams
getting a fifth-kill (gets five kill first)
are 2.7 and 1.5, respectively. Assuming the
probability of each team getting a kill is 50%,
should I place a bet?

"""
import random


def simulate_once(prob):
    a_prob = prob

    kill_list = []
    a_kill, b_kill = 0, 1
    while a_kill < 5 and b_kill < 5:
        if random.random() < a_prob:
            a_kill += 1
        else:
            b_kill += 1

        kill_list.append((a_kill, b_kill))

    return a_kill > b_kill, kill_list


def has_reversal(kill_list: list):
    for kills in kill_list:
        a_kill, b_kill = kills
        if a_kill > b_kill:
            return True
    return False

def simulate_five_kill(prob=0.5,
                       n=1000,
                       init_money=10000,
                       bet_money=10,
                       odds=2.8,
                       ):
    """
    prob: The probability of team A to get a kill.
    n: Number of simulations
    """
    money = init_money
    r_money = init_money
    win_num = 0
    reverse_num = 0
    for _ in range(n):
        win, kill_list = simulate_once(prob)
        if win:
            win_num += 1
            money += (odds - 1) * bet_money
        else:
            money -= bet_money

        if has_reversal(kill_list):
            reverse_num += 1
            r_money += (odds - 2) * bet_money
        else:
            r_money -= bet_money

    win_rate = win_num / n * 100

    print("初始金额:", init_money)
    print("每次投注金额:", bet_money)
    print("总投注次数:", n)
    print("赔率:", odds)
    print("胜率:", win_rate)
    print("反转率:", reverse_num)
    print("最终金额(反转投注):", r_money)
    print("最终金额(一次投注):", money)


if __name__ == '__main__':
    simulate_five_kill(prob=0.55,
                       odds=2.7,
                       init_money=1000,
                       bet_money=50,
                       n=100)

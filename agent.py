import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.misc
import os

from environment import Environment



class Agent():
    # 代理状态配置的值数
    STATE_DIM = 2  # 持股比例，投资组合价值比例

    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)

    TRADING_CHARGE = 0  # 0.00015
    TRADING_TAX = 0  # 0.003

    def __init__(self, environment, min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.05):

        self.environment = environment

        # 最小交易单位、最大交易单位、延迟补偿阈值
        self.min_trading_unit = min_trading_unit  # 最小单交易单位
        self.max_trading_unit = max_trading_unit
        self.delayed_reward_threshold = delayed_reward_threshold  # 延迟补偿阈值

        # Agent 类的属性
        self.initial_balance = 1000000  # 初期资本金
        self.balance = 0  # 当前现金余额
        self.num_stocks = 0  # 持有股份数
        self.portfolio_value = 0  # balance + num_stocks * {当前股票价格}
        self.base_portfolio_value = 0  # 前一个
        self.num_buy = 0  # 收购次数
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0  # 即时奖励

        self.profit_rate_account = []  # 账号盈利

        # Agent 类的状态
        self.ratio_hold = 0  # 持股比例
        self.ratio_portfolio_value = 0  # 投资组合价值百分比



    def reset(self):
        self.maximum_portfolio_value = self.initial_balance
        self.minimum_portfolio_value = self.initial_balance
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value  #每日资产净值
        return [
            self.ratio_hold,
            self.ratio_portfolio_value
        ]


    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.
        # 探险决定
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)  # 随机决定行动
        else:
            exploration = False
            probs = policy_network.predict(sample)  # 每个行动的概率
            action = np.argmax(probs)
            confidence = probs[action]
        return action, confidence, exploration

    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:

            if self.balance < self.environment.get_price() * (
                        1 + self.TRADING_CHARGE) * self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:

            if self.num_stocks <= 0:
                validity = False
        return validity

    # confidence : 选定行为的概率值
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit - self.min_trading_unit
        ), 0)
        #added_traiding=100
        return self.min_trading_unit + added_traiding

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 获取环境中的当前价格
        curr_price = self.environment.get_price()  ##股票价格

        # 立即重置奖励
        self.immediate_reward = 0

        # 买入
        if action == Agent.ACTION_BUY:
            # 确定要购买的单位
            trading_unit = self.decide_trading_unit(confidence)
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # 如果持有的现金不足，尽可能用持有的现金购买
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (
                        curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                    self.min_trading_unit
                )
            # 应用手续费计算总买入金额
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  # 刷新持有现金
            self.num_stocks += trading_unit
            self.num_buy += 1  # 매수 횟수 증가
            #self.states_buy.append(self.environment.idx)

        # 卖出
        elif action == Agent.ACTION_SELL:
            # 确定要出售的单位
            trading_unit = self.decide_trading_unit(confidence)
            # 如果持有的股票不足，尽可能最大限度地抛售
            trading_unit = min(trading_unit, self.num_stocks)
            # 抛售
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit
            self.balance += invest_amount
            self.num_sell += 1
            #self.states_sell.append(self.environment.idx)

        # 保持
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 持球次数增加

        # 产品组合价值更新
        self.portfolio_value = self.balance + curr_price * self.num_stocks  #组合价值，收益，用这个计算日回报率
        if self.maximum_portfolio_value < self.portfolio_value:
            self.maximum_portfolio_value = self.portfolio_value
        elif self.minimum_portfolio_value > self.portfolio_value:
            self.minimum_portfolio_value = self.portfolio_value

        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)*100

        #self.profit_rate_account.append(profitloss)


            #利润增长率  今日净值增长率=（今日净值－昨日净值）/昨日净值
        #print(profitloss)
        #self.base_portfolio_value = self.portfolio_value


        return profitloss#self.immediate_reward
        #return profitloss  #self.immediate_reward, delayed_reward

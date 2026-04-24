import MonteCarloSimulator as m
import Backtester as b
import MarketEnvironment as me
import AlmgrenChrissModel as ac

def main():

    strategy = ac.AlmgrenChrissModel(375, 30, 10, 40, 55, 0.05, 0.1)
    env = me.MarketEnvironment(375, 30, 10, 40, 55, 0.05)

    sim = m.MonteCarloSimulator(375, 30, 10, 40, 55, 0.05, 0.1)
    back = b.Backtester(strategy_model=strategy, market_env=env)

    print(f'Result of run_single_lambda: {sim.run_single_lambda(0.5)}')
    print(f'Result from run_lambda_grid: {sim.run_lambda_grid([0.3,0.4,0.5,0.6])}')

    print(f'Result of running backtester: {back.run()}')


if __name__ == "__main__":
    main()

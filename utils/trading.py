from regex import D


class Portfolio:
    def __init__(self, start_date) -> None:
        self.pnl = 0
        self.pnl_dict = {start_date: self.pnl}
        self.positions = {'BTC-USD': 0, 'ETH-USD': 0}

    def insert_order(self, instrument_id:str, side:str, price:float, volume:float) -> None:
        assert side in ['BUY', 'SELL']
        sign = -1 if side == 'SELL' else 1
        self.positions[instrument_id] += sign * volume
        self.pnl -= sign * volume * price

    def close_positions(self, btc_close:float, eth_close:float, date:str) -> None:
        self.update_pnl(btc_close, eth_close, date)
        self.positions = {key: 0 for key in self.positions.keys()}

    def get_positions_value(self, btc_close:float, eth_close:float) -> float:
        return self.positions['BTC-USD'] * btc_close + self.positions['ETH-USD'] * eth_close

    def update_pnl(self, btc_close:float, eth_close:float, date:str) -> None:
        self.pnl += self.get_positions_value(btc_close, eth_close)
        self.pnl_dict[date] = self.pnl

    def get_last_pnl(self) -> float:
        return self.pnl

    def get_pnl_dict(self) -> list[float]:
        return self.pnl_dict
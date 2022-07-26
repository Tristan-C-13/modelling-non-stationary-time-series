class Portfolio:
    def __init__(self, btc_position_0, eth_position_0) -> None:
        self.pnl = 0
        self.btc_position = btc_position_0
        self.eth_position = eth_position_0

    def insert_order(instrument_id:str, side:str, volume:float):
        pass

    def get_positions(self) -> dict[str, int]: # {btc: 10, eth: 36} 
        pass

    def get_pnl(self) -> float:
        pass
from src.find_stock import PickStock


class OverfallPickStock(PickStock):

    def buy_or_not(self, reader: TuShareDataReader) -> bool:
        pass
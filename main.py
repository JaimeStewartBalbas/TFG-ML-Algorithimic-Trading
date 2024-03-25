from DataRetrieval import HistoricalDataRetrieval

if __name__ == '__main__':
    yearly_data_retrieval = HistoricalDataRetrieval('Ibex 35','^IBEX',365)
    yearly_data_retrieval.plot_data()

    monthly_data_retrieval = HistoricalDataRetrieval('Ibex 35', '^IBEX',30)
    monthly_data_retrieval.plot_data()

    weekly_data_retrieval = HistoricalDataRetrieval('Ibex 35','^IBEX',7)
    weekly_data_retrieval.plot_data()

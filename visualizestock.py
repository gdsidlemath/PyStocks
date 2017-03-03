import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime
import warnings
#import urllib.request

warnings.filterwarnings("ignore")

aa = np.asarray


def getsymbolsector():
    NYSEurl = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download'

    NASDAQurl = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download'

    AMEXurl = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=AMEX&render=download'

    NYSE = pd.read_csv(NYSEurl, error_bad_lines=False)

    NASDAQ = pd.read_csv(NASDAQurl, error_bad_lines=False)

    AMEX = pd.read_csv(AMEXurl, error_bad_lines=False)

    NYSE = NYSE[NYSE.Sector != 'n/a']

    NASDAQ = NASDAQ[NASDAQ['Sector'] != 'n/a']

    AMEX = AMEX[AMEX['Sector'] != 'n/a']

    AllStocks = pd.concat([NYSE, NASDAQ, AMEX], keys=['NYSE', 'NASDAQ', 'AMEX'])

    MegaCap = np.where(AllStocks['MarketCap'] > 2e11)[0]
    BigCap = np.intersect1d(np.where(AllStocks['MarketCap'] <= 2e11)[0], np.where(AllStocks['MarketCap'] > 1e10)[0])
    MidCap = np.intersect1d(np.where(AllStocks['MarketCap'] <= 1e10)[0], np.where(AllStocks['MarketCap'] > 2e9)[0])
    SmallCap = np.intersect1d(np.where(AllStocks['MarketCap'] <= 2e9)[0], np.where(AllStocks['MarketCap'] > 3e8)[0])
    MicroCap = np.intersect1d(np.where(AllStocks['MarketCap'] <= 3e8)[0], np.where(AllStocks['MarketCap'] > 5e7)[0])
    NanoCap = np.where(AllStocks['MarketCap'] <= 5e7)[0]

    AllStocks.MarketCap[MegaCap] = 1
    AllStocks.MarketCap[BigCap] = 2
    AllStocks.MarketCap[MidCap] = 3
    AllStocks.MarketCap[SmallCap] = 4
    AllStocks.MarketCap[MicroCap] = 5
    AllStocks.MarketCap[NanoCap] = 6

    # print(AllStocks[['Symbol','Sector','MarketCap']])

    return AllStocks[['Symbol', 'Sector', 'MarketCap']]


def visualizereturn(symbol, yrrange,tm):

    AllStocks = getsymbolsector().reset_index(drop=True)
    
    Close,x = getClosingData(symbol,yrrange)
    
    x = np.atleast_2d(x).T

    fk = len(Close)

    Cend = aa(Close[1:])
    Cbeg = aa(Close[:fk-1])

    sinds = np.where(AllStocks.Symbol.isin([symbol]))
    try:
        sector = AllStocks.Sector.iloc[int(sinds[0])]
        print(sector)
    except:
        sector = 'Fund or ETF'

    BenClose = benchmarkreturn(yrrange, fk)
    
    try:
    
        AllStocks.drop(AllStocks.index[int(sinds[0])])

        secInds = np.where(AllStocks.Sector.isin([sector]))[0]

        SizInds = np.where(AllStocks.MarketCap.iloc[secInds].isin([1]))[0]#np.hstack((np.where(AllStocks.MarketCap.iloc[secInds].isin([5]))[0],np.where(AllStocks.MarketCap.iloc[secInds].isin([6]))[0]))

        secSymb = AllStocks.Symbol.iloc[secInds[SizInds]]

        SecClose = np.zeros([fk, len(SizInds)])

        for ind,symb in enumerate(secSymb):

            try:
                SecClose[:, ind] = getClosingData(symb,yrrange)[0]
            except:
                continue

        SecClose = SecClose[:, np.where(SecClose[0, :] != 0)[0]]
        SecDaily = np.vstack((np.zeros([1,np.shape(SecClose)[1]]),np.divide((SecClose[1:,:] - SecClose[:(fk-1),:]), SecClose[:(fk-1),:])*100))
        
    except:
        SecClose = np.ones([1,fk])
        SecDaily = np.zeros([1,fk])

    BenDaily = np.vstack((np.zeros([1,np.shape(BenClose)[1]]),np.divide((BenClose[1:,:] - BenClose[:(fk-1),:]), BenClose[:(fk-1),:])*100))

    tf1 = int(tm[0])
    tf2 = int(tm[1])
    tf3 = int(tm[2])
    
    temp = np.divide((Cend - Cbeg),Cbeg)

    stockdaily = np.hstack((aa([0]),temp*100))
    return10day = SReturn(Close, tf1)  # np.zeros([len(test['Adj Close']),1])
    return20day = SReturn(Close, tf2)
    return40day = SReturn(Close, tf3)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(x, stockdaily, color='r', label=symbol + ' Return')
    #ax.plot(x, np.mean(SecDaily, axis=1), color='b', label='Sector MegaCap Return')
    ax.plot(x, BenDaily[:,0], color='g', label='S&P 500 Return')
    ax.set_ylabel('Percent Return')
    ax.set_title('Daily Return')
    #ax.legend()

    ax1 = fig.add_subplot(222)
    ax1.plot(x, return10day, color='r', label=symbol + ' Return')
    #ax1.plot(x, SReturn(SecClose, tf1), color='b', label='Sector MegaCap Return')
    ax1.plot(x, SReturn(BenClose[:,0], tf1), color='g', label='S&P 500 Return')
    ax1.set_ylabel('Percent Return')
    ax1.set_title(str(tf1) + ' Day Return')
    #ax1.legend()

    ax2 = fig.add_subplot(223)
    ax2.plot(x, return20day, color='r', label=symbol + ' Return')
    #ax2.plot(x, SReturn(SecClose, tf2), color='b', label='Sector MegaCap Return')
    ax2.plot(x, SReturn(BenClose[:, 0], tf2), color='g', label='S&P 500 Return')
    ax2.set_ylabel('Percent Return')
    ax2.set_title(str(tf2) + ' Day Return')
    #ax2.legend()

    ax3 = fig.add_subplot(224)
    ax3.plot(x, return40day, color='r', label=symbol + ' Return')
    #ax3.plot(x, SReturn(SecClose, tf3), color='b', label='Sector MegaCap Return')
    ax3.plot(x, SReturn(BenClose[:, 0], tf3), color='g', label='S&P 500 Return')
    ax3.set_ylabel('Percent Return')
    ax3.set_title(str(tf3) + ' Day Return')
    #ax3.legend()

    if 'Fund' in sector:
        print('Correlation between Fund and S&P Return: ' + str(np.corrcoef(SReturn(Close,fk).T,SReturn(BenClose[:, 0],fk).T)[1,0]))
    else:
        ax.plot(x, np.mean(SecDaily, axis=1), color='b', label='Sector MegaCap Return')
        ax1.plot(x, SReturn(SecClose, tf1).T, color='b', label='Sector MegaCap Return')
        ax2.plot(x, SReturn(SecClose, tf2).T, color='b', label='Sector MegaCap Return')
        ax3.plot(x, SReturn(SecClose, tf3).T, color='b', label='Sector MegaCap Return')
        print('Correlation between Stock and Sector Return: ' + str(np.corrcoef(SReturn(Close,fk).T,SReturn(SecClose,fk))[1,0]))

    ax.legend(loc=0, borderpad=0.5, labelspacing=0.5,fontsize=8)
    ax1.legend(loc=0, borderpad=0.5, labelspacing=0.5,fontsize=8)
    ax2.legend(loc=0, borderpad=0.5, labelspacing=0.5,fontsize=8)
    ax3.legend(loc=0, borderpad=0.5, labelspacing=0.5,fontsize=8)

    fig2 = plt.figure()
    bx = fig2.add_subplot(111)
    bx.plot(x,SReturn(BenClose[:,0], fk), color='b', label='S&P')
    bx.plot(x,SReturn(BenClose[:,1], fk), color='g', label='Dow')
    bx.plot(x,SReturn(BenClose[:,2], fk), color='r', label='NASDAQ')
    bx.legend(loc=0, borderpad=0.5, labelspacing=0.5,fontsize=8)

    plt.show()
    
    return 0

def getClosingData(symbol,yrrange):

    now = datetime.datetime.now()
    
    day = str(now.day - 1)
    month = str(now.month - 1)
    yr = str(now.year - yrrange)
    
    url = 'http://chart.finance.yahoo.com/table.csv?s=' + symbol + '&a=' + month + '&b=' + day + '&c=' + yr + '&d=' + month + '&e=' + day + '&f=' + str(now.year) + '&g=d&ignore=.csv'
                                                                                                                                                        
    #s = urllib.request.urlopen(url)
                                                                                                                                                        
    test = pd.read_csv(url)
                                                                                                                                                        
    test = test[::-1]
    
    x = pd.to_datetime(test.Date)
                                                                                                                                                        
    Close = aa(test['Adj Close'])

    return Close, x

def SReturn(Close, tf):
    if Close.ndim > 1:
        ret = np.zeros([np.shape(Close)[0], np.shape(Close)[1]])
        for i in range(1, np.shape(Close)[0]):
            if i < (tf - 1):
                ret[i, :] = np.divide((Close[i, :] - Close[0, :]), Close[0, :])
            else:
                ret[i, :] = np.divide((Close[i, :] - Close[i - (tf - 1), :]), Close[i - (tf - 1), :])
        out = np.mean(ret, axis=1)
    else:
        ret = np.zeros([len(Close), 1])
        for i in range(1, len(Close)):
            if i < (tf - 1):
                ret[i] = (Close[i] - Close[0])/Close[0]
            else:
                ret[i] = (Close[i] - Close[i - (tf - 1)])/Close[i - (tf - 1)]
        out = ret
    return np.atleast_2d(out)*100

def benchmarkreturn(yrrange,ln):

    now = datetime.datetime.now()

    day = str(now.day - 1)
    month = str(now.month - 1)
    yr = str(now.year - yrrange)

    BenSymb = ['^GSPC','^DJI','^IXIC']

    BenClose = np.zeros([ln,3])

    for i,symb in enumerate(BenSymb):

        BenClose[:, i] = getClosingData(symb,yrrange)[0]

    return BenClose

symbol = raw_input('What stock do you want to look at? ').upper()

yrrange = int(raw_input('How far back? '))

tm = [raw_input('Timeframe 1? '), raw_input('Timeframe 2? '), raw_input('Timeframe 3? ')]

visualizereturn(symbol, yrrange, tm)

# getsymbolsector()


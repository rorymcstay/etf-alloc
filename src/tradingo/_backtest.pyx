cimport numpy as np
import numpy as np


def compute_backtest(
    double opening_position,
    double opening_avg_price,
    double[:] trades,
    double[:] prices,
):

    cdef int num_days = trades.shape[0]
    cdef int idx, idx_prev

    cdef double[:] unrealised_pnl = np.zeros(num_days)
    cdef double[:] realised_pnl = np.zeros(num_days)
    cdef double[:] total_pnl = np.zeros(num_days)
    cdef double[:] net_investment = np.zeros(num_days)
    cdef double[:] net_position = np.zeros(num_days)
    cdef double[:] avg_open_price = np.zeros(num_days)

    net_position[0] = opening_position
    avg_open_price[0] = opening_avg_price

    cdef double m_net_position = opening_position
    cdef double m_avg_open_price = opening_avg_price
    cdef double m_unrealised_pnl = 0
    cdef double m_total_pnl = 0
    cdef double m_realised_pnl = 0
    cdef double m_net_investment = 0

    for idx in range(1, num_days):

        idx_prev = idx - 1

        trade_quantity = trades[idx]
        price = prices[idx]
        m_unrealised_pnl = (price - avg_open_price[idx_prev]) * net_position[idx_prev]

        m_net_position = net_position[idx_prev]
        m_realised_pnl = realised_pnl[idx_prev]
        m_avg_open_price = avg_open_price[idx_prev]

        if trade_quantity != 0 and not np.isnan(trade_quantity):

            # net investment
            m_net_investment = max(
                net_investment[idx_prev],
                abs(net_position[idx_prev] * avg_open_price[idx_prev])
            )
            # realized pnl
            if abs(net_position[idx_prev] + trade_quantity) < abs(net_position[idx_prev]):
                m_realised_pnl += (
                    (price - m_avg_open_price) * abs(trade_quantity) * np.sign(net_position[idx_prev])
                )
            # total pnl
            m_total_pnl = m_realised_pnl + m_unrealised_pnl
            # avg open price
            if abs(net_position[idx_prev] + trade_quantity) > abs(net_position[idx_prev]):
                m_avg_open_price = (
                    (m_avg_open_price * m_net_position) + (price * trade_quantity)
                ) / (net_position[idx_prev] + trade_quantity)
            else:
                # Check if it is close-and-open
                if trade_quantity > abs(net_position[idx_prev]):
                    m_avg_open_price = price

        m_net_position += trade_quantity
        m_unrealised_pnl = (price - m_avg_open_price) * net_position[idx_prev]

        unrealised_pnl[idx] = m_unrealised_pnl
        realised_pnl[idx] = m_realised_pnl
        total_pnl[idx] = m_total_pnl
        net_investment[idx] = m_net_investment
        net_position[idx] = m_net_position
        avg_open_price[idx] = m_avg_open_price

    return np.column_stack(
        (
            np.asarray(unrealised_pnl),
            np.asarray(realised_pnl),
            np.asarray(total_pnl),
            np.asarray(net_investment),
            np.asarray(net_position),
            np.asarray(avg_open_price),
        )
    )

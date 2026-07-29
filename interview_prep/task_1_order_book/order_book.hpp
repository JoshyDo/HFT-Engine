#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace imc_prep {

enum class Side {
    BUY,
    SELL
};

struct Execution {
    uint64_t maker_order_id;
    uint64_t taker_order_id;
    uint32_t price;
    uint32_t qty;
};

class OrderBookListener {
public:
    virtual ~OrderBookListener() = default;
    virtual void on_trade(const Execution& exec) = 0;
};

class OrderBook {
public:
    explicit OrderBook(OrderBookListener* listener) : listener_(listener) {}

    void add_order(uint64_t id, Side side, uint32_t price, uint32_t qty) {
        if (side == Side::BUY) {
            while (qty > 0 && !asks_.empty() && price >= get_best_ask()) {
                auto ask_it = asks_.begin();
                uint32_t best_ask_price = ask_it->first;
                auto& orders_at_price = ask_it->second;

                auto& resting_order = orders_at_price.front();
                uint32_t trade_qty = std::min(qty, resting_order.qty);

                listener_->on_trade({
                    resting_order.id,
                    id,
                    best_ask_price,
                    trade_qty
                });

                qty -= trade_qty;
                resting_order.qty -= trade_qty;

                if (resting_order.qty == 0) {
                    order_locations_.erase(resting_order.id);
                    orders_at_price.pop_front();
                }

                if (orders_at_price.empty()) {
                    asks_.erase(ask_it);
                }
            }

            if (qty > 0) {
                bids_[price].push_back({id, side, price, qty});
                auto it = std::prev(bids_[price].end());
                order_locations_[id] = it;
            }
        } else {
            while (qty > 0 && !bids_.empty() && price <= get_best_bid()) {
                auto bid_it = bids_.begin();
                uint32_t best_bid_price = bid_it->first;
                auto& orders_at_price = bid_it->second;

                auto& resting_order = orders_at_price.front();
                uint32_t trade_qty = std::min(qty, resting_order.qty);

                listener_->on_trade({
                    resting_order.id,
                    id,
                    best_bid_price,
                    trade_qty
                });

                qty -= trade_qty;
                resting_order.qty -= trade_qty;

                if (resting_order.qty == 0) {
                    order_locations_.erase(resting_order.id);
                    orders_at_price.pop_front();
                }

                if (orders_at_price.empty()) {
                    bids_.erase(bid_it);
                }
            }

            if (qty > 0) {
                asks_[price].push_back({id, side, price, qty});
                auto it = std::prev(asks_[price].end());
                order_locations_[id] = it;
            }
        }
    }

    void cancel_order(uint64_t id) {
        auto it = order_locations_.find(id);
        if (it == order_locations_.end()) return;
        
        auto list_it = it->second; 
        
        if (list_it->side == Side::BUY) {
            bids_[list_it->price].erase(list_it);
            if (bids_[list_it->price].empty()) {
                bids_.erase(list_it->price);
            }
        } else {
            asks_[list_it->price].erase(list_it);
            if (asks_[list_it->price].empty()) {
                asks_.erase(list_it->price);
            }
        }

        order_locations_.erase(it);
    }

    uint32_t get_best_bid() const {
        if (bids_.empty()) return 0;
        return bids_.begin()->first;
    }

    uint32_t get_best_ask() const {
        if (asks_.empty()) return 0;
        return asks_.begin()->first;
    }

private:
    struct Order {
        uint64_t id;
        Side side; 
        uint32_t price;
        uint32_t qty;
    };

    OrderBookListener* listener_;
    
    std::map<uint32_t, std::list<Order>, std::greater<uint32_t>> bids_;
    std::map<uint32_t, std::list<Order>, std::less<uint32_t>> asks_;
    std::unordered_map<uint64_t, std::list<Order>::iterator> order_locations_;
};

} // namespace imc_prep

package com.leetcode.easy;

/**
 * easy题集
 * @author lihao on 2023/1/11
 */
public class Solution {

    /**
     * 121. 买卖股票的最佳时机 https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/
     * 思路: 先求出当前index卖出可以获取的利润,再更新买入值(即nums[index]最小值)
     */
    public int maxProfit(int[] prices) {
        int low = prices[0];
        int max = 0;
        for(int i=0;i<prices.length;i++) {
            max = Math.max(max,prices[i]-low);
            low = Math.min(low,prices[i]);
        }
        return max;
    }
}

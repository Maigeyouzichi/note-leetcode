package com.leetcode.easy;

/**
 * easy题集
 * @author lihao on 2023/1/11
 */
@SuppressWarnings("all")
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

    /**
     * 9. Palindrome Number, 回文数字 https://leetcode.com/problems/palindrome-number/
     * 思路: 不断取其最后一个数字构建新的数字,原来的数字不断舍弃低位,如果新数字大于原来的数字,判断两个数字是不是相等或者差一个地位数字区别
     * 例子: 12321 或者 1221, 最后比较 12 和 123, 123/10 == 12 或者 12 == 12 的情况都认为是满足条件的.
     */
    public boolean isPalindrome(int x) {
        if(x == 0) return true;
        if(x < 0 || x%10 == 0) return false;
        int reverse = 0;
        while(x > reverse) {
            reverse = reverse*10 + x%10;
            x /= 10;
        }
        return x == reverse || x == reverse/10;
    }
}

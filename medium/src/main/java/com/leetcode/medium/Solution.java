package com.leetcode.medium;

import com.leetcode.base.ListNode;
import java.util.Stack;
import com.leetcode.base.TreeNode;
/**
 * @author lihao on 2022/12/26
 */
public class Solution {

    /**
     * 2,两数相加 https://leetcode.cn/problems/add-two-numbers/
     * 思路: 遍历两个链表,不存在的节点值作0处理,依次相加,进位值单独存储,最后单独处理进位值
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode();
        ListNode curr = head;
        //保存进位数
        int k = 0;
        while(l1 != null || l2 != null) {
            int first = l1 == null ? 0 : l1.val;
            int second = l2 == null ? 0 : l2.val;
            int val = first + second + k;
            curr.next = new ListNode(val%10);
            curr = curr.next;
            k = val/10;
            l1 = l1==null?null:l1.next;
            l2 = l2==null?null:l2.next;
        }
        if(k>0) curr.next = new ListNode(k);
        return head.next;
    }

    /**
     * 3. 无重复字符的最长子串 https://leetcode.cn/problems/longest-substring-without-repeating-characters/
     * 思路:滑动窗口,右指针先滑,遍历后标记,如果遇到重复的,左指针滑动重复元素右一个元素.
     * tips:
     *  虽然题目说明由英文字母、数字、符号和空格组成,但是0~255就够了
     *  字符表示值范围: a-z: 97-122  A-Z: 65-90
     */
    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0 || s.length() == 1) return s.length();
        char[] arr = s.toCharArray();
        int left = 0,right = 0;
        boolean[] bitArr = new boolean[256];
        int rns = 0;
        while(right < arr.length) {
            if(bitArr[arr[right]]) {
                while(arr[left] != arr[right]) { bitArr[arr[left++]] = false; }
                left++;
            }
            bitArr[arr[right]] = true;
            rns = Math.max(rns,right-left+1);
            right++;
        }
        return rns;
    }

    /**
     * 5. 最长回文子串 https://leetcode.cn/problems/longest-palindromic-substring/
     * 思路: 动态规划,dp[i][i]一定为true,双重for循环,两个指针,如果指针元素相同且相邻,则dp[][]=true,
     *  如果不相邻且dp[+1][-1]=true,则dp[][]=true,如此,利用动态规划即可.
     */
    public String longestPalindrome(String s) {
        char[] charArray = s.toCharArray();
        boolean[][] dp = new boolean[s.length()][s.length()];
        int left = 0, right = 0;
        for (int i = 0; i < s.length(); i++) {
            dp[i][i] = true;
        }
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (charArray[i] != charArray[j]) { continue; }
                if (i - j == 1 || (i - j > 1 && dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                    if (i - j > right - left) {
                        left = j;
                        right = i;
                    }
                }
            }
        }
        return s.substring(left, right + 1);
    }


    /**
     * 11. 盛最多水的容器 https://leetcode.cn/problems/container-with-most-water/
     * 思路: 双指针,容器盛水的多少,取决于最短的板,每次滑动最短的板即可
     */
    public int maxArea(int[] height) {
        int left = 0,right = height.length-1;
        int rns = 0;
        while(left < right) {
            int volume = Math.min(height[left],height[right])*(right-left);
            rns = Math.max(volume,rns);
            if(height[left] > height[right]) {
                int preRight = right;
                while(height[preRight]>= height[right] && left<right) right--;
            }else {
                int preLeft = left;
                while(height[preLeft]>=height[left] && left<right) left++;
            }
        }
        return rns;
    }


    /**
     * 补充: 二叉树的后续遍历,非递归解法 https://mp.weixin.qq.com/s/mBXfpH4nuIltyHm72zLryw
     */
    public static void postOrder(TreeNode tree) {
        if (tree == null) return;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(tree);
        TreeNode c;
        while (!stack.isEmpty()) {
            c = stack.peek();
            if (c.left != null && tree != c.left && tree != c.right) {
                stack.push(c.left);
            } else if (c.right != null && tree != c.right) {
                stack.push(c.right);
            } else {
                System.out.printf(stack.pop().val + "");
                tree = c;
            }
        }
    }


}

package com.leetcode.medium;

import com.leetcode.base.ListNode;
import com.leetcode.base.TreeNode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;
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

    /**
     * 三数之和 https://leetcode.cn/problems/3sum/
     * 思路: 排序+三指针,最先确定左指针,从0开始,遇到相同的跳过,进而把问题转换成双指针问题,left,right指针相向而行,遇到相同的值进行跳过
     * 复杂度: N²
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums == null || nums.length < 3) return result;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length-2; i++) {
            if (nums[i] > 0) return result;
            //答案中不可以包括相同的三元组
            if (i >0 && nums[i] == nums[i-1]) continue;
            int left = i +1;
            int right = nums.length -1;
            while (left < right) {
                int sum = nums[i]+nums[left]+nums[right];
                if (sum == 0) {
                    result.add(Arrays.asList(nums[i],nums[left],nums[right]));
                    //答案中不可以包括相同的三元组
                    while (left<right && nums[left] == nums[left+1]) left ++;
                    while (right > left && nums[right] == nums[right -1]) right --;
                    left ++;
                    right --;
                }else if (sum > 0) {
                    right --;
                }else {
                    left ++;
                }
            }
        }
        return result;
    }

    /**
     * 16. 最接近的三数之和 https://leetcode.cn/problems/3sum-closest/
     * 思路: 和三数之和一样,遍历每一个元素,三指针转换成双指针问题,相对于三数之和,没有去重跳过的问题,反而编码上更简单
     * 吐槽: 同样的代码,第一次提交是5ms,现在是13ms,不知道力扣的服务器发生了什么
     */
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int result = nums[0]+nums[1]+nums[2];
        for (int i = 0; i < nums.length-2; i++) {
            int left = i+1;
            int right = nums.length -1;
            while (left < right) {
                int sum = nums[i]+nums[left]+nums[right];
                if (Math.abs(target-result)>Math.abs(target-sum)) {
                    result = sum;
                }
                if (sum == target) {
                    return sum;
                } else if (sum > target) {
                    right --;
                } else {
                    left ++;
                }
            }
        }
        return result;
    }

}

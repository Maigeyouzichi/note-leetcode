package com.leetcode.hard;

import com.leetcode.base.ListNode;

@SuppressWarnings("all")
class Solution {

    /**
     * 面试题 17.19. 消失的两个数字 https://leetcode.cn/problems/missing-two-lcci/
     * 思路: 1,找到缺失的两个值的和missSum
     *      2,其中一个缺失的数字一定在[1,missSum/2]之间
     *      3,计算[1,missSum/2]的和减去原数组中小于等于missSum/2的数字的结果就是第一个缺失的数字
     */
    public int[] missingTwo(int[] nums) {
        int originalLen = nums.length + 2;
        int originalSum = (1 + originalLen) * originalLen / 2;
        int missSum = originalSum;
        for (int num : nums) { missSum -= num; }
        int edgeNum = missSum / 2;
        int originalSumOfEdge = (1 + edgeNum) * edgeNum / 2;
        int missOne = originalSumOfEdge;
        for (int num : nums) {
            if (num > edgeNum) { continue; }
            missOne -= num;
        }
        return new int[]{missOne, missSum - missOne};
    }

    /**
     * 同上,空间复杂度O(n),上述解法满足题目要求的空间复杂度O(1)
     * 思路: 1~n可以全部放入长度为n+1的数组中,放入元素标记一下,没被标记的就是缺失的数字.
     */
    public int[] missingTwo_(int[] nums) {
        int[] rns = new int[2];
        int index = 0;
        boolean[] bitArr = new boolean[nums.length+3];
        for(int nu: nums) {
            bitArr[nu] = true;
        }
        for(int i=1;i<nums.length+3;i++) {
            if(bitArr[i]) continue;
            rns[index++] = i;
            if(index == 2) break;
        }
        return rns;
    }

    /**
     * 23. 合并K个升序链表 https://leetcode.cn/problems/merge-k-sorted-lists/
     * 思路: 两两合并,转换成合并2个升序数组的问题
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) {
            return null;
        }
        ListNode rns = lists[0];
        for (int i = 1; i < lists.length; i++) {
            ListNode curr = lists[i];
            rns = mergeListNode(rns, curr);
        }
        return rns;
    }

    private ListNode mergeListNode(ListNode left, ListNode right) {
        ListNode head = new ListNode();
        ListNode curr = head;
        while (left != null || right != null) {
            if (left == null) {
                curr.next = right;
                break;
            }
            if (right == null) {
                curr.next = left;
                break;
            }
            if (left.val > right.val) {
                curr.next = right;
                right = right.next;
            } else {
                curr.next = left;
                left = left.next;
            }
            curr = curr.next;
        }
        return head.next;
    }
}
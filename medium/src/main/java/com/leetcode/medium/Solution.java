package com.leetcode.medium;

import com.leetcode.base.DisJointSet;
import com.leetcode.base.ListNode;
import com.leetcode.base.TreeNode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
/**
 * @author lihao on 2022/12/26
 */
@SuppressWarnings("all")
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

    /**
     * 19. 删除链表的倒数第 N 个结点  https://leetcode.cn/problems/remove-nth-node-from-end-of-list/
     * 思路: 两次遍历
     *  第一次遍历: 获取链表总长度,并获取目标节点的index(index从0开始)
     *  第二次遍历: 考虑不同情况不同处理,A-目标节点为头节点,B-目标节点为尾节点,C-目标节点为中间节点
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode tmpNode = head;
        int sz = 1;
        while (tmpNode.next != null) {
            tmpNode = tmpNode.next;
            sz ++;
        }
        int index = sz - n;
        //如果是目标节点是第一个节点
        if (index == 0) { return head.next;}
        tmpNode = head;
        for (int i = 0; i < index-1; i++) {
            tmpNode = tmpNode.next;
        }
        if (index == sz-1) {
            //如果是目标节点是最后一个节点
            tmpNode.next = null;
        }else {
            //如果是目标节点是中间节点
            tmpNode.next = tmpNode.next.next;
        }
        return head;
    }

    /**
     * 22. 括号生成 https://leetcode.cn/problems/generate-parentheses/
     * 思路: 动态规划,dp[n]表示i对括弧可以生成的括号组合,n=0表示空串,n对括弧可以由"( m对括弧的组合 ) + n-1-m对括弧的组合" 组合而成
     */
    public List<String> generateParenthesis(int n) {
        //dp[i]表示i对括弧可以生成的括号组合,i=0表示空串
        List<List<String>> dp = new ArrayList<>();
        //设置初始数据,增加空串,为了兼容n=1的场景
        List<String> origin = new ArrayList<>();
        origin.add("");
        dp.add(origin);
        for(int i=1;i<n+1;i++) {
            List<String> currList = new ArrayList<>();
            for(int j=0;j<i;j++) {
                for(String m: dp.get(j)) {
                    for(String k: dp.get(i-1-j)) {
                        currList.add("("+m+")"+k);
                    }
                }
            }
            dp.add(currList);
        }
        return dp.get(n);
    }

    /**
     * 24. 两两交换链表中的节点 https://leetcode.cn/problems/swap-nodes-in-pairs/
     * 思路: 递归, swapPairs()方法将传入的node节点和后节点交换并返回后节点.
     */
    public ListNode swapPairs(ListNode head) {
        //方法的解释: 输入node节点,将node和next node交换后返回前面的节点
        if (head == null || head.next == null) { return head; }
        ListNode next = head.next;
        head.next = swapPairs(next.next);
        next.next = head;
        return next;
    }

    /**
     * 28. 找出字符串中第一个匹配项的下标 https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/
     * 思路: 经典kmp算法 todo 待理解
     */
    public int strStr(String haystack, String needle) {
        //构建next数组
        int m = haystack.length();
        int n = needle.length();
        if (n==0) return 0;
        if (n>m) return -1;
        int[] next = new int[n];
        int j = 0;
        for (int i = 1; i < n; i++) {
            while (j>0 && needle.charAt(i) != needle.charAt(j)) {
                j = next[j-1];
            }
            if (needle.charAt(i) == needle.charAt(j)) {
                j ++;
            }
            next[i] = j;
        }
        //遍历 -> 判断子串
        j = 0;
        for (int i = 0; i < m; i++) {
            while (j>0 && haystack.charAt(i) != needle.charAt(j)) {
                j = next[j-1];
            }
            if (needle.charAt(j) == haystack.charAt(i)) {
                j ++;
            }
            if (j == n) return i-(n-1);
        }
        return -1;
    }

    /**
     * 31. 下一个排列 https://leetcode.cn/problems/next-permutation/
     * 思路: 其实应该叫下一个字典更大排序更合适,下一个更大的话,优先动低位的元素,这样才是下一个更大.
     *  1,先考虑一般情况,字典序更大的组合,就是把尽可能低位的元素大的替换小的,如果从右向左都是递增的,那么断然是没有符合条件的替换场景的,
     *  找到第一个值下降的元素,index记作i,需要被比它更大的元素替换,从右向左找第一个比它更大的元素,索引记作j
     *  nums[i]和nums[j]交换后,将i之后的元素升序排序即可(交换后i之后元素刚好是递减的,因此两两交换就可以了)
     *  2,再考虑典型情况,比如: [3,2,1] 或者 [3],增加兼容逻辑
     */
    public void nextPermutation(int[] nums) {// 1, 2, 3, 4, 5, 5, 4, 3, 2, 1
        int i = nums.length - 2;
        //从右向左找第一次下降的数字
        while (i > 0) {
            if (nums[i]>=nums[i+1]) { i --; }else { break; }
        }
        //兼容只要一个元素的场景,比如: [3]
        i = Math.max(i, 0);
        int j = nums.length -1;
        //从右向左找第一个比nums[i]大的数字
        while (j > i) {
            if (nums[j] <= nums[i]){ j --; }else { break; }
        }
        swap(nums,i,j);
        //兼容没有下一个排列的情况,比如: 3,2,1,0
        if (i ==0 && j ==0) {
            i --;
        }
        reverseNodeList(nums,i+1,nums.length-1);
    }

    /**
     * 根据index两两交换元素
     */
    private void reverseNodeList(int[] nums, int i, int j) {
        while (i < j) {
            swap(nums,i, j);
            i ++;
            j --;
        }
    }

    /**
     * 根据index交换数组元素
     */
    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    /**
     * 33. 搜索旋转排序数组 https://leetcode.cn/problems/search-in-rotated-sorted-array/
     * 思路: 二分查找
     *  题目要求复杂度: O(log n),可以自然的想到二分查找,由于nums本身是连续的,被"旋转"后仍然是分段有序,找到分段的index,
     *  分别进行二分即可
     */
    public int search(int[] nums, int target) {
        int index = 0;
        for(int i=1;i<nums.length;i++) {
            if(nums[i]-nums[i-1]<0) {
                index = i;
                break;
            }
        }
        int indexLeft = binarySearch(nums,0,index-1,target);
        if(indexLeft > -1) return indexLeft;
        int indexRight = binarySearch(nums,index,nums.length-1,target);
        if(indexRight > -1) return indexRight;
        return -1;
    }

    /**
     * 二分查找
     */
    private int binarySearch(int[] nums,int start, int end, int target) {
        while(start<=end) {
            int mid = (start+end)/2;
            if(nums[mid] > target) {
                end = mid-1;
            }else if(nums[mid] < target) {
                start = mid+1;
            }else {
                return mid;
            }
        }
        return -1;
    }

    /**
     * 34. 在排序数组中查找元素的第一个和最后一个位置
     * https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/
     * 思路: 二分查找
     *  两次二分,分别找出target的index的最小值和最大值,二分的实现当arr[mid] == target的时候进行移动下标即可.
     */
    public int[] searchRange(int[] nums, int target) {
        int indexLeft = binarySearchInCondition(nums,target,true);
        int indexRight = binarySearchInCondition(nums,target, false);
        return new int[]{indexLeft,indexRight};
    }

    /**
     * 有条件的二分查找
     * 存在多个结果,根据传入条件,返回目标值最小index或者最大index
     */
    int binarySearchInCondition(int[] arr, int target, boolean leftFlag) {
        if(arr.length == 0) return -1;
        int left = 0,right = arr.length-1;
        int res = -1;
        while(left<=right) {
            int mid = left+(right-left)/2;
            if(arr[mid] > target) {
                right = mid - 1;
            }else if(arr[mid]< target) {
                left = mid + 1;
            }else {
                res = mid;
                if(leftFlag) { right = mid -1; }else { left = mid+1;}
            }
        }
        return res;
    }

    /**
     * 39. 组合总和 https://leetcode.cn/problems/combination-sum/
     * 思路: 回溯 -- 解决组合问题一般都是回溯 todo 待理解
     * 重复元素的组合,每次元素使用后,还要从当前元素开始重复相同的过程,使用for循环是从前往后遍历,这里使用递归,可以实现从后往前遍历
     * 1,target初始值的时候,从第一个元素递归到最后,依次弹栈,等于是从后往前进行遍历
     * 2,每处理一次循环,会新add一个元素,这时target就会减少,然后重新跑一遍1的流程.
     * 3,递归中套着递归,最后最外层的栈全部弹出,执行结束
     */
    List<List<Integer>> rns = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        backTracing(candidates,target,0);
        return rns;
    }

    /**
     * 函数定义: 给定源数组,目标值,初始下标,将符合条件的组合加入到结果集合中
     * @param candidates 源数组
     * @param target 目标值
     * @param idx 下标
     */
    private void backTracing(int[] candidates,int target,int idx) {
        if(idx == candidates.length) { return; }
        if(target == 0) {
            rns.add(new ArrayList<>(path));
            return;
        }
        //从后往前遍历
        backTracing(candidates,target,idx+1);
        if(target >= candidates[idx]) {
            path.add(candidates[idx]);
            //target变了之后也要将整个过程再重复一遍
            backTracing(candidates,target - candidates[idx],idx);
            path.removeLast();
        }
    }

    /**
     * 45. 跳跃游戏 II https://leetcode.cn/problems/jump-game-ii/
     * 思路: 从第0个各自开始算段落,第0个能到达的最远的index是下一个边界,移动到下一个边界的时候确定下一个边界,
     * 每次经过边界的时候,step++,最后符合条件的时候,step+1返回.
     */
    public int jump(int[] nums) {
        if (nums[0] == 0 || nums.length ==1) return 0;
        int step = 0, currentMaxIndex = 0,stepMaxIndex = 0;
        for (int i = 0; i < nums.length; i++) {
            currentMaxIndex = Math.max(currentMaxIndex,i+nums[i]);
            if (currentMaxIndex>=nums.length-1) {
                return step+1;
            }
            if (i == stepMaxIndex) {
                step ++;
                stepMaxIndex = currentMaxIndex;
            }
        }
        //一定会提前返回
        return 0;
    }

    /**
     * 46. 全排列 https://leetcode.cn/problems/permutations/
     * 思路: 回溯
     * 每一次递归都遍历全部的元素,path中如果有重复的就跳过本次循环
     */
    public List<List<Integer>> permute(int[] nums) {
        backTracing(nums);
        return rns;
    }

    private void backTracing(int[] nums) {
        if (path.size() == nums.length) {
            rns.add(new ArrayList<>(path));
            return;
        }
        for (int nu : nums) {
            if (path.contains(nu)) { continue; }
            path.add(nu);
            backTracing(nums);
            path.removeLast();
        }
    }

    /**
     * 53. 最大子数组和 https://leetcode.cn/problems/maximum-subarray/
     * 思路: 因为是连续的子数组,遍历一次,pre记录当前最大值,pre+nums[i] 和 nums[i]比较去较大值,同时更新res即可
     */
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        int pre = nums[0];
        for(int i=1;i<nums.length;i++) {
            pre = Math.max(pre+nums[i],nums[i]);
            res = Math.max(res,pre);
        }
        return res;
    }

    /**
     * 55. 跳跃游戏 https://leetcode.cn/problems/jump-game/
     * 思路:不用考虑步数,所以只需要考虑最大可到达index即可,指针向右滑动,同时更新maxIndex
     */
    public boolean canJump(int[] nums) {
        int currentIndex = 0, maxIndex = 0;
        while(currentIndex <= maxIndex) {
            maxIndex = Math.max(currentIndex+nums[currentIndex],maxIndex);
            if(maxIndex >= nums.length-1) return true;
            currentIndex++;
        }
        return false;
    }

    /**
     * 56. 合并区间 https://leetcode.cn/problems/merge-intervals/
     * 思路: 优先队列
     * 借助优先队列,将时间块按照startTime进行排序,依次遍历,根据next[0]和current[0]比较进行分支判断
     */
    public int[][] merge(int[][] intervals) {
        PriorityQueue<int[]> needVisitIntervals = new PriorityQueue<>((o1,o2) -> o1[0] - o2[0]);
        for(int[] interval : intervals) {
            needVisitIntervals.offer(interval);
        }
        List<int[]> mergedIntervals = new ArrayList<>();
        int[] needCompareInterval = needVisitIntervals.poll();
        while(!needVisitIntervals.isEmpty()) {
            int[] currInterval = needVisitIntervals.poll();
            if(needCompareInterval[1] >= currInterval[0]) {
                needCompareInterval[1] = Math.max(needCompareInterval[1],currInterval[1]);
            }else {
                mergedIntervals.add(needCompareInterval);
                needCompareInterval = currInterval;
            }
        }
        mergedIntervals.add(needCompareInterval);
        return mergedIntervals.toArray(new int[mergedIntervals.size()][]);
    }

    /**
     * 61. 旋转链表 https://leetcode.cn/problems/rotate-list/
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null) return head;
        ListNode tmpNode = head;
        //求出链表总长度
        int nodeLength = 1;
        while (tmpNode.next != null) {
            tmpNode = tmpNode.next;
            nodeLength ++;
        }
        //链表变成了环
        tmpNode.next = head;
        //找到新的头结点,并和前一个节点断开
        k = k % nodeLength;
        for (int i = 0; i < nodeLength-k-1; i++) {
            head = head.next;
        }
        ListNode tmp = head;
        head = head.next;
        tmp.next = null;
        return head;
    }

    /**
     * 62. 不同路径 https://leetcode.cn/problems/unique-paths/
     * 思路: 动态规划
     * do[i][j]表示机器人走到(从0开始)为i,j的位置,一共有多少不同的走法.
     */
    public int uniquePaths(int m, int n) {
        //动态规划
        int[][] dp = new int[m][n];
        for(int i=0;i<m;i++) {
            dp[i][0] = 1;
        }
        for(int i=0;i<n;i++) {
            dp[0][i] = 1;
        }
        for(int i=1;i<m;i++) {
            for(int j=1;j<n;j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }

    /**
     * 63. 不同路径 II https://leetcode.cn/problems/unique-paths-ii/
     * 思路: 动态规划
     * do[i][j]表示机器人走到(从0开始)为i,j的位置,一共有多少不同的走法.有障碍物的格子,dp[i][j]=0
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1) {
                break;
            } else {
                dp[i][0] = 1;
            }
        }
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 1) {
                break;
            } else {
                dp[0][i] = 1;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * 64. 最小路径和 https://leetcode.cn/problems/minimum-path-sum/
     * 思路: 动态规划
     * dp[i][j]含义是到达位置(i,j)的最小路径和
     */
    public int minPathSum(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        dp[0][0] = grid[0][0];
        for(int i=1;i<grid.length;i++) {
            dp[i][0] = dp[i-1][0]+grid[i][0];
        }
        for(int i=1;i<grid[0].length;i++) {
            dp[0][i] = dp[0][i-1]+grid[0][i];
        }
        for(int i=1;i<grid.length;i++) {
            for(int j=1;j<grid[0].length;j++) {
                dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1]) + grid[i][j];
            }
        }
        return dp[grid.length-1][grid[0].length-1];
    }

    /**
     * 75. 颜色分类 https://leetcode.cn/problems/sort-colors/
     * 思路: 交换
     * 条件比较特殊,只存在三种不同的数字,没有必要使用排序算法进行排序,两次遍历,第一次将所有的0放在前面,第二次将所有的1放在前面即可.
     */
    public void sortColors(int[] nums) {
        //p永远指向需要被交换到后面的元素
        int p = 0;
        int tmp = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                tmp = nums[i];
                nums[i] = nums[p];
                nums[p] = tmp;
                p ++;
            }
        }
        for (int i = p; i < nums.length; i++) {
            if (nums[i] == 1) {
                tmp = nums[i];
                nums[i] = nums[p];
                nums[p] = tmp;
                p ++;
            }
        }
    }

    /**
     * 77. 组合 https://leetcode.cn/problems/combinations/
     * 思路: 标准回溯
     * 循环中套着递归
     */
    public List<List<Integer>> combine(int n, int k) {
        backTracking(1, n, k);
        return rns;
    }

    private void backTracking(int startIndex,int endIndex,int k) {
        //剪枝 降低时间复杂度 执行时间从15ms -> 2ms
        if (path.size()+endIndex-startIndex+1 < k) {
            return;
        }
        if (path.size() == k) {
            rns.add(new ArrayList<>(path));
            return;
        }
        for (int i = startIndex; i <= endIndex; i++) {
            path.add(i);
            backTracking(i+1, endIndex, k);
            path.removeLast();
        }
    }

    /**
     * 78. 子集 https://leetcode.cn/problems/subsets/
     * 思路: 和上面的组合一样,区别在于k有多重情况
     */
    public List<List<Integer>> subsets(int[] nums) {
        for(int k= 0; k<= nums.length;k++) {
            backTracing(nums,k,0, nums.length-1);
        }
        return rns;
    }

    private void backTracing(int[] nums, int k, int startIndex, int endIndex) {
        //剪枝
        if (path.size()+endIndex-startIndex+1 < k) return;
        if (path.size() == k) {
            rns.add(new ArrayList<>(path));
            return;
        }
        for (int i = startIndex; i <= endIndex; i++) {
            path.add(nums[i]);
            backTracing(nums, k, i + 1, endIndex);
            path.removeLast();
        }
    }

    /**
     * 82. 删除排序链表中的重复元素 II
     * 思路: 1,虚拟一个头结点 2,遍历所有节点,判断当前节点是否为有效节点,有效则连接 3,最后将最后一个节点断尾
     */
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null || head.next == null) return head;
        boolean[] bitArr = new boolean[201];
        ListNode virtualValidHeadNode = new ListNode();
        ListNode validNode = virtualValidHeadNode;
        ListNode currentNode = head;
        while(currentNode != null) {
            if(!bitArr[currentNode.val+100] && (currentNode.next == null || currentNode.val != currentNode.next.val)) {
                validNode.next = currentNode;
                validNode = validNode.next;
            }
            bitArr[currentNode.val+100] = true;
            currentNode = currentNode.next;
        }
        validNode.next = null;
        return virtualValidHeadNode.next;
    }

    /**
     * 86. 分隔链表 https://leetcode.cn/problems/partition-list/
     * 例子: 1,4,3,2,5,2 x=3
     * 思路: 移动
     * 题目要求保持原来的相对位置,所以不能直接进行重新赋值的操作,先找到第一个小于target的节点,后面遇到小于target的值就移动到左边接续的位置.
     */
    public ListNode partition(ListNode head, int x) {
        int target = x;
        if(head == null) return head;
        //创建虚拟头节点
        ListNode virturalHeadNode = new ListNode();
        virturalHeadNode.next = head;
        //声明过程变量
        ListNode currentEdgeNode = virturalHeadNode, currentRightNode = head.next, currentLeftNode = head;
        while(currentEdgeNode.next != null) {
            if(currentEdgeNode.next.val >= target) break;
            currentEdgeNode = currentEdgeNode.next;
        }
        while(currentRightNode != null) {
            if(currentLeftNode.val >= target && currentRightNode.val < target) {
                currentLeftNode.next = currentRightNode.next;
                currentRightNode.next = currentEdgeNode.next;
                currentEdgeNode.next = currentRightNode;
                //交换后遍历到下一组节点
                currentRightNode = currentLeftNode.next;
                currentEdgeNode = currentEdgeNode.next;
            }else {
                currentRightNode = currentRightNode.next;
                currentLeftNode = currentLeftNode.next;
            }
        }
        return virturalHeadNode.next;
    }

    /**
     * 128. 最长连续序列 https://leetcode.cn/problems/longest-consecutive-sequence/
     * 例子: [100,4,200,1,3,2]
     * 思路: HashMap key:num value:length
     * 每次都去更新上下边缘num对应的value,中间的num不会影响最大length
     * 进阶: 并查集
     */
    public int longestConsecutive(int[] nums) {
        Map<Integer,Integer> numWithLen = new HashMap<>();
        int maxLen = 0,currLen = 0;
        for(int num: nums) {
            if(numWithLen.containsKey(num)) continue;
            int leftLen = numWithLen.getOrDefault(num-1,0);
            int right = numWithLen.getOrDefault(num+1,0);
            currLen = leftLen+right+1;
            maxLen = Math.max(maxLen,currLen);
            numWithLen.put(num,currLen);
            numWithLen.put(num-leftLen,currLen);
            numWithLen.put(num+right,currLen);
        }
        return maxLen;
    }

    /**
     * 137. 只出现一次的数字 II https://leetcode.cn/problems/single-number-ii/
     * 思路: 用一个数组记录元素每一位的数字和,结果对3取模的值就是目标数字所在位的值,二进制转十进制即可.
     */
    public int singleNumber(int[] nums) {
        //将每个数的每个位的数字进行相加,如果能整除3,说明目标数当前位为0,否则说明目标数当前位为1,通过目标数每个位的数字求出其十进制表示
        int[] bitArray = new int[32];
        for(int nu: nums) {
            for(int i = 0; i<32; i++) {
                int bit = (nu >> i)&1;
                bitArray[31-i]+=bit;
            }
        }
        int res = 0;
        for(int i = 0; i< 32; i++) {
            res += (bitArray[i]%3)<<(31-i);
        }
        return res;
    }

    /**
     * 142. 环形链表 II https://leetcode.cn/problems/linked-list-cycle-ii/
     * 思路: 快慢指针
     * 不成环的节点有a个, 成环的节点有b个, 相遇时距离环的距离是x个
     * 相遇时快指针路径: a + m * b + x
     * 相遇时慢指针路径: a + n * b + x
     * (m-2*n) * b = a + x  -> k * b - x = a 即: 围着环转b圈再往回倒x个节点,刚好等于a个节点,所以相遇的时候快指针从head开始即可.
     *
     */
    public ListNode detectCycle(ListNode head) {
        if(head == null) return null;
        ListNode fast = head;
        ListNode slow = head;
        while(true) {
            fast = fast.next;
            slow = slow.next;
            //防止空指针
            if(fast== null || slow == null) return null;
            fast = fast.next;
            if(fast== null || slow == null) return null;
            if(fast == slow) break;
        }
        //快指针回到head,每次走一个格子,再次相遇就是成环的地方
        fast = head;
        while(fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }

    /**
     * 148. 排序链表 https://leetcode.cn/problems/sort-list/
     * 思路: 分治算法
     * 一个长链表分割成两个小链表,小链表继续分割,最终分割成单个链表,再两两有序合并.
     */
    public ListNode sortList(ListNode head) {
        return sortAndMerge(head);
    }

    private ListNode sortAndMerge(ListNode node) {
        //校验
        if(node ==null ||node.next == null) return node;
        //分割链表
        ListNode binaryNode = binaryNode(node);
        ListNode left = sortAndMerge(node);
        ListNode right = sortAndMerge(binaryNode);
        //合并链表
        return mergeListNode(left,right);
    }

    //快慢指针找到中间的节点并分割链表
    private ListNode binaryNode(ListNode node) {
        ListNode fast = node;
        ListNode slow = node;
        while(fast.next!=null && fast.next.next!=null) {
            fast = fast.next;
            fast = fast.next;
            slow = slow.next;
        }
        ListNode binaryNode = slow.next;
        slow.next = null;
        return binaryNode;
    }

    //合并两个链表
    private ListNode mergeListNode(ListNode left,ListNode right) {
        ListNode head = new ListNode();
        ListNode curr = head;
        while(left!=null || right!=null) {
            if(left==null) {
                curr.next = right;
                break;
            }
            if(right == null) {
                curr.next = left;
                break;
            }
            if(left.val > right.val) {
                curr.next = right;
                right = right.next;
            }else{
                curr.next = left;
                left = left.next;
            }
            curr = curr.next;
        }
        return head.next;
    }

    /**
     * 150. 逆波兰表达式求值 https://leetcode.cn/problems/evaluate-reverse-polish-notation/
     * 思路: stack
     */
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        int numA = 0;
        int numB = 0;
        for(String str: tokens) {
            if(str.equals("+")) {
                numA = stack.pop();
                numB = stack.pop();
                stack.push(numB+numA);
            }else if(str.equals("-")) {
                numA = stack.pop();
                numB = stack.pop();
                stack.push(numB-numA);
            }else if(str.equals("*")) {
                numA = stack.pop();
                numB = stack.pop();
                stack.push(numB*numA);
            }else if(str.equals("/")) {
                numA = stack.pop();
                numB = stack.pop();
                stack.push(numB/numA);
            }else {
                stack.push(Integer.valueOf(str));
            }
        }
        return stack.pop();
    }

    /**
     * 152. 乘积最大子数组 https://leetcode.cn/problems/maximum-product-subarray/
     * 思路: 遍历
     * 乘积最大无非要考虑两个因素,一个是0的情况,一个是结果为负数的情况
     * 1, 如果是0,就变成1,这样结果是合理的
     * 2, 前后各跑一遍,就避免了结果为负数导致的情况,比如: 第一个数字是-1,后面都是正数,从前遍历结果为负数,从后遍历结果就是正数
     */
    public int maxProduct(int[] nums) {
        int rnsLeft = 0, rnsRight = 0 , rns = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (rnsLeft == 0) { rnsLeft = 1; }
            rnsLeft *= nums[i];
            rns = Math.max(rns, rnsLeft);
            if (rnsRight == 0) { rnsRight = 1; }
            rnsRight *= nums[nums.length - 1 - i];
            rns = Math.max(rns, rnsRight);
        }
        return rns;
    }

    /**
     * 167. 两数之和 II - 输入有序数组 https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/
     * 思路: 双指针
     * 两个指针相向遍历,找到目标index
     */
    public int[] twoSum(int[] numbers, int target) {
        int left = 0, right = numbers.length -1;
        while (left < right) {
            int sum = numbers[left]+numbers[right];
            if (sum == target) {
                return new int[]{left+1,right+1};
            }else if (sum > target) {
                right --;
            }else {
                left ++;
            }
        }
        return null;
    }

    /**
     * 189. 轮转数组 https://leetcode.cn/problems/rotate-array/
     * 思路: 复制数组
     * 找到边界下标,复制数组,重新为原数组赋值
     */
    public void rotate(int[] nums, int k) {
        int[] copyArray = Arrays.copyOfRange(nums, 0, nums.length);
        int edgeIndex = nums.length - k % (nums.length), currentIndex = 0;
        for (int i = edgeIndex; i < nums.length; i++) {
            nums[currentIndex] = copyArray[i];
            currentIndex++;
        }
        for (int i = 0; i < edgeIndex; i++) {
            nums[currentIndex] = copyArray[i];
            currentIndex++;
        }
    }

    /**
     * 198. 打家劫舍 https://leetcode.cn/problems/house-robber/
     * 思路: 动态规划
     * dp[i]表示前i+1个房子可以偷到的财产最大总和,前两家单独讨论.
     */
    public int rob(int[] nums) {
        if(nums.length == 1) return nums[0];
        if(nums.length == 2) return Math.max(nums[0],nums[1]);
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0],nums[1]);
        for(int i=2;i<nums.length;i++) {
            dp[i] = Math.max(dp[i-2]+nums[i],dp[i-1]);
        }
        return dp[nums.length-1];
    }

    /**
     * 207. 课程表 https://leetcode.cn/problems/course-schedule/
     * 思路: 拓扑排序
     * 0,全部课程(节点)会生成一张有向无环图 1,准备邻接表(某个node的全部后续node) 2,准备入度表 3,准备队列,存储满足遍历条件的node 4,入度为0即可加入队列 5,最后判断是否存在未遍历到的节点
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        //本质上还是判断是否存在环
        //拓扑排序--BFS
        //邻接表
        HashSet<Integer>[] adjacencyTable = new HashSet[numCourses];
        //入度表
        int[] inDegreeTable = new int[numCourses];
        //返回序列省略 .. 只需要判断是否存在环即可,不需要得到拓扑序列
        Queue<Integer> visitedQueue = new LinkedList<>();//没有检索用LinkedList,存在检索用ArrayDeque
        int pollCount = 0;
        for(int i=0;i<numCourses;i++) {
            adjacencyTable[i] = new HashSet<>();
        }
        for(int[] degree: prerequisites) {
            inDegreeTable[degree[0]]++;
            adjacencyTable[degree[1]].add(degree[0]);
        }
        for(int i=0;i<numCourses;i++) {
            if(inDegreeTable[i] == 0) visitedQueue.offer(i);
        }
        while(!visitedQueue.isEmpty()) {
            int currentNode = visitedQueue.poll();
            pollCount++;
            for(int nextNode : adjacencyTable[currentNode]) {
                inDegreeTable[nextNode]--;
                if(inDegreeTable[nextNode]==0) visitedQueue.offer(nextNode);
            }
        }
        return pollCount == numCourses;
    }

    /**
     * 210. 课程表 II https://leetcode.cn/problems/course-schedule-ii/
     * 思路: 拓扑排序
     * 和上面的课程表题目一样, 只不过多出了结果的输出步骤.
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        HashSet<Integer>[] adjacencyArr = new HashSet[numCourses];
        int[] inDegreeArr = new int[numCourses];
        Queue<Integer> queue = new LinkedList<>();
        int[] rns = new int[numCourses];
        int index = 0;
        //遍历接续关系集合,构建邻接数组和入度数组
        for(int i=0;i<numCourses;i++) {
            adjacencyArr[i] = new HashSet<>();
        }
        for(int[] edge: prerequisites) {
            adjacencyArr[edge[1]].add(edge[0]);
            inDegreeArr[edge[0]]++;
        }
        //遍历邻接表,将入度为0的结果加入队列
        for(int i=0;i<numCourses;i++) {
            if(inDegreeArr[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            Integer head = queue.poll();
            rns[index++] = head;
            HashSet<Integer> nodes = adjacencyArr[head];
            for(int node: nodes) {
                inDegreeArr[node]--;
                if(inDegreeArr[node]==0) { queue.offer(node); }
            }
        }
        if(index != numCourses) return new int[0];
        return rns;
    }

    /**
     * 213. 打家劫舍 II https://leetcode.cn/problems/house-robber-ii/
     * 思路: 动态规划
     * 和打家劫舍区别在于房子是成环的,考虑第一个房子就不考虑最后一个房子,考虑最后一个房子就不考虑第一个房子
     */
    public int rob2(int[] nums) {
        int len = nums.length;
        if(len == 1) return nums[0];
        if(len == 2) return Math.max(nums[0],nums[1]);
        return Math.max(invokeRob(nums,0,len-2), invokeRob(nums,1,len-1));
    }

    //分开考虑 0~n-1 和 1~n 的结果
    private int invokeRob(int[] nums, int startIndex, int endIndex) {
        int[] dp = new int[nums.length-1];
        dp[0] = nums[startIndex];
        dp[1] = Math.max(nums[startIndex],nums[startIndex+1]);
        for(int i= 2;i<endIndex-startIndex+1;i++) {
            dp[i] = Math.max(dp[i-2]+nums[startIndex+i],dp[i-1]);
        }
        return dp[nums.length-2];
    }

    /**
     * 215. 数组中的第K个最大元素 https://leetcode.cn/problems/kth-largest-element-in-an-array/submissions/
     * 思路: 计数排序
     * 使用数组记录元素的值和数量,然后从后往前遍历即可.
     */
    public int findKthLargest(int[] nums, int k) {
        int[] countArr = new int[20001];
        for(int num: nums) { countArr[num+10000]++; }
        for(int i= countArr.length-1;i>=0;i--) {
            k -= countArr[i];
            if(k <= 0) return i-10000;
        }
        return 0;
    }

    /**
     * 237. 删除链表中的节点 https://leetcode.cn/problems/delete-node-in-a-linked-list/
     * 思路: 题目约定要删除的节点不是末尾节点,可以将下个节点的值赋予当期节点,同时将next节点剔除当前链表中.
     */
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    /**
     * 260. 只出现一次的数字 III https://leetcode.cn/problems/single-number-iii/
     * 思路: 二进制,所有数字相同的位的数字进行异或,结果不为0的位可以作为判断标准去将元素分组,当前位为0的一组,为1的一组,分别进行异或,得到结果
     */
    public int[] singleNumber3(int[] nums) {
        int xor = 0 , index = 0;
        for(int i=0;i< 32;i++) {
            for(int nu: nums) {
                xor ^= (nu >> i & 1);
            }
            if(xor != 0) {
                index = i;
                break;
            }
        }
        int rns_a = 0, rns_b = 0;
        for(int num: nums) {
            if(((num>>index)&1) == 0) {
                rns_a^= num;
            }else {
                rns_b ^= num;
            }
        }
        return new int[]{rns_a,rns_b};
    }

    /**
     * 264. 丑数 II https://leetcode.cn/problems/ugly-number-ii/
     * 思路: 优先队列
     * 从1开始,每次出队列的时候,当前元素分别乘以1,3,5并发放入队列,无限循环,直到第n个数从队列中取出
     * tips: 1,防止溢出,使用Long型 2,使用Set防止重复数据加入队列
     */
    public int nthUglyNumber(int n) {
        PriorityQueue<Long> pq = new PriorityQueue<>();
        Set<Long> set = new HashSet<>();
        Long[] arr = {2L,3L,5L};
        pq.offer(1L);
        int count = 0;
        while(!pq.isEmpty()) {
            Long curr = pq.poll();
            count ++;
            if(count == n) return Math.toIntExact(curr);
            for(Long num: arr) {
                Long tmp = curr*num;
                if(set.add(tmp)) pq.offer(tmp);
            }
        }
        return 0;
    }

    /**
     * 287. 寻找重复数 https://leetcode.cn/problems/find-the-duplicate-number/
     * 思路: 快慢指针(龟兔赛跑)
     * 类似环形链表的解法,有两个index指向同一个value,所以链表一定是成环的
     */
    public int findDuplicate(int[] nums) {
        int fast = 0;
        int slow = 0;
        while(true) {
            fast = nums[nums[fast]];
            slow = nums[slow];
            if(fast == slow) {
                fast = 0;
                while(fast != slow) {
                    fast = nums[fast];
                    slow = nums[slow];
                }
                return slow;
            }
        }
    }

    /**
     * 322. 零钱兑换 https://leetcode.cn/problems/coin-change/
     * 思路: 动态规划,完全背包问题
     * dp[i]表示组成和为i需要的最少的硬币的个数,coins[i]最小是1,可以将dp数组中全部设为count+1
     * dp[i] = Math.min(dp[i],dp[i-coin]+1)
     */
    public int coinChange(int[] coins, int amount) {
        //dp[i]表示组成和为i需要的最少的硬币的个数
        int max = amount+1;
        int[] dp = new int[amount+1];
        Arrays.fill(dp,max);
        dp[0] = 0;
        for(int i=1;i<amount+1;i++) {
            for(int val: coins) {
                if(val > i) continue;
                dp[i] = Math.min(dp[i],dp[i-val]+1);
            }
        }
        return dp[amount]>amount?-1:dp[amount];
    }

    /**
     * 343. 整数拆分 https://leetcode.cn/problems/integer-break/
     * 思路: 数学定理(当n大于4开始,拆成足够多的3即可)
     */
    public int integerBreak(int n) {
        if(n < 4) {
            return n-1;
        }
        int res = 1;
        while(n > 4) {
            res *= 3;
            n -= 3;
        }
        return res*n;
    }

    /**
     * 347. 前 K 个高频元素 https://leetcode.cn/problems/top-k-frequent-elements/
     * 思路: 小顶堆
     */
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();
        for(int num: nums) {
            map.put(num,map.getOrDefault(num,0)+1);
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>((o1,o2) -> map.get(o1)-map.get(o2));
        for(Map.Entry<Integer,Integer> entry: map.entrySet()) {
            pq.offer(entry.getKey());
            if(pq.size()>k) pq.poll();
        }
        int[] rns = new int[k];
        for(int i=0;i<k;i++) {
            rns[i] = pq.poll();
        }
        return rns;
    }

    /**
     * 376. 摆动序列 https://leetcode.cn/problems/wiggle-subsequence/
     * 思路: 遍历
     * 两两相减结果写入数组,遍历数组正负相同跳过,记录正负不同的组数,得出结果
     */
    public int wiggleMaxLength(int[] nums) {
        if(nums.length == 1) return 1;
        int[] diffArr = new int[nums.length];
        int left = 0, right = 1, index = 0;
        while(right < nums.length) {
            if(nums[right] != nums[left]) {
                diffArr[index++] = nums[right]-nums[left];
                left = right;
            }
            right++;
        }
        if(diffArr[0] == 0) return 1;
        int count = 0;
        left = 0;
        right = 1;
        while(right < diffArr.length) {
            if(diffArr[right]==0) break;
            if(diffArr[left]*diffArr[right]<0) {
                count++;
                left = right;
            }
            right++;
        }
        return count+2;
    }

    /**
     * 445. 两数相加 II https://leetcode.cn/problems/add-two-numbers-ii/
     * 思路: 分别进行翻转链表 -> 计算 -> 再次对结果进行翻转链表
     */
    public ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
        ListNode node1 = reverseNodeList(l1);
        ListNode node2 = reverseNodeList(l2);
        ListNode rns = new ListNode();
        ListNode tmp = rns;
        int k = 0;
        while(node1!= null || node2 != null) {
            int num1 = node1 == null? 0: node1.val;
            int num2 = node2 == null? 0: node2.val;
            int val = (num1+num2+k)%10;
            k = (num1+num2+k)/10;
            tmp.next = new ListNode(val);
            tmp = tmp.next;
            if(node1 != null) node1 = node1.next;
            if(node2 != null) node2 = node2.next;
        }
        if(k>0) {
            tmp.next = new ListNode(k);
        }
        return reverseNodeList(rns.next);
    }

    /**
     * 翻转链表
     */
    private ListNode reverseNodeList(ListNode node) {
        ListNode pre = null;
        ListNode curr = node;
        ListNode next = node.next;
        while(curr != null) {
            next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }

    /**
     * 453. 最小操作次数使数组元素相等 https://leetcode.cn/problems/minimum-moves-to-equal-array-elements/
     * 思路: 数学
     * 每次n-1个元素都+1 等同于 每次1个元素-1, 这样所有的元素都减至和最小值一样即可
     */
    public int minMoves(int[] nums) {
        int min = nums[0];
        for (int i = 1; i < nums.length; i++) {
            min = Math.min(min,nums[i]);
        }
        int res = 0;
        for (int num : nums) {
            res += num - min;
        }
        return res;
    }

    /**
     * 494. 目标和 https://leetcode.cn/problems/target-sum/
     * 思路: todo 有时间再想吧
     */
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for(int num: nums) sum += num;
        int temp = sum + target;//这里写 sum-target 也是可以的
        if(temp < 0 || (temp&1)==1) return 0;
        int packageSize = temp/2;
        //dp[i][j]表示前i个数,可以组成和为j的方案数
        int[][] dp = new int[nums.length+1][packageSize+1];
        dp[0][0] = 1;
        for(int i=1;i<=packageSize;i++) dp[0][i] = 0;
        for(int i=1;i<=nums.length;i++) {
            for(int j=0;j<=packageSize;j++) {
                if(nums[i-1]>j) {
                    dp[i][j] = dp[i-1][j];
                }else {
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]];
                }
            }
        }
        return dp[nums.length][packageSize];
    }

    /**
     * 547. 省份数量 https://leetcode.cn/problems/number-of-provinces/
     * 思路: 并查集
     * 每个城市节点对应其最上层的父结点,最终父结点的总数就是要求的结果
     */
    public int findCircleNum(int[][] isConnected) {
        DisJointSet jointSet = new DisJointSet(isConnected.length);
        for (int i = 0; i < isConnected.length; i++) {
            for (int j = i + 1; j < isConnected.length; j++) {
                if (isConnected[i][j] == 0) {
                    continue;
                }
                jointSet.union(i, j);
            }
        }
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < isConnected.length; i++) {
            set.add(jointSet.findRoot(i));
        }
        return set.size();
    }

    /**
     * 560. 和为 K 的子数组 https://leetcode.cn/problems/subarray-sum-equals-k/
     * 思路: 一次遍历,前缀和以及出现的次数放入map,0-1提前放入map,如果prefix_sum - k的结果也出现在数组中,count+=结果即可
     */
    public int subarraySum(int[] nums, int k) {
        //key:前缀和 value:次数
        Map<Integer, Integer> map = new HashMap<>();
        //为了包含index从0开始的子数组,前缀和刚好等于k的时候,count++
        map.put(0, 1);
        int count = 0, prefixSum = 0;
        for (int nu : nums) {
            prefixSum += nu;
            if (map.containsKey(prefixSum - k)) {
                count += map.get(prefixSum - k);
            }
            map.put(prefixSum, map.getOrDefault(prefixSum, 0) + 1);
        }
        return count;
    }

    /**
     * 649. Dota2 参议院 https://leetcode.cn/problems/dota2-senate/
     * 思路: 贪心算法
     * 博弈逻辑:每次按照顺序就近投死对手,投完准备下一轮投票,优先投后面的,后面没有就投前面的
     * 准备两个队列,分别对应天辉和夜魇,如果没出局投完票继续加入队列
     */
    public String predictPartyVictory(String senate) {
        int len = senate.length();
        char[] charArray = senate.toCharArray();
        LinkedList<Integer> queue_D = new LinkedList<>();
        LinkedList<Integer> queue_R = new LinkedList<>();
        for (int i = 0; i < len; i++) {
            if (charArray[i] == 'D') {
                queue_D.offer(i);
            } else {
                queue_R.offer(i);
            }
        }
        while (!queue_R.isEmpty() || !queue_D.isEmpty()) {
            if (queue_R.isEmpty()) {
                return "Dire";
            }
            if (queue_D.isEmpty()) {
                return "Radiant";
            }
            if (queue_D.peek() < queue_R.peek()) {
                queue_D.offer(queue_D.poll() + len);
                queue_R.poll();
            } else {
                queue_R.offer(queue_R.poll() + len);
                queue_D.poll();
            }
        }
        return "";
    }

    /**
     * 684. 冗余连接 https://leetcode.cn/problems/redundant-connection/
     * 思路: 并查集
     * 遍历的过程填充并查集,当遍历到一组节点已经是属于一组了,直接返回,如果存在多个结果,当前返回也是最后出现的组合.
     */
    public int[] findRedundantConnection(int[][] edges) {
        DisJointSet dis = new DisJointSet(edges.length);
        for(int[] arr: edges) {
            int x = arr[0];
            int y = arr[1];
            if(dis.findRoot(x)==dis.findRoot(y)) return arr;
            dis.union(x,y);
        }
        return edges[edges.length-1];
    }

    /**
     * 720. 词典中最长的单词 https://leetcode.cn/problems/longest-word-in-dictionary/
     * 遍历
     * 1,所有元素放入set 2,words进行自然排序 3,遍历数组,当前元素取子串判断
     */
    public String longestWord(String[] words) {
        Set<String> set = new HashSet<>();
        //所有单词写入set
        for(int i=0;i<words.length;i++){
            set.add(words[i]);
        }
        int maxLen = 0;
        String res = "";
        //排序
        Arrays.sort(words);
        //遍历排序后的数组
        for(int i=0;i<words.length;i++){
            //如果该字符串长度小于res,直接继续,因为结果是取最长的单词
            if(words[i].length() <= maxLen) {
                continue;
            }
            boolean flag = true;
            //判断该单词是否符合要求: 由数组中单词组成
            for(int j=1;j<words[i].length();j++){
                flag = set.contains(words[i].substring(0,j));
                if(!flag) break;
            }
            //由于是排好序的,因此结果会取字典索引最小的: 该字典等同于自然排序
            if(flag) {
                maxLen = words[i].length();
                res = words[i];
            }
        }
        return res;
    }

    /**
     * 739. 每日温度 https://leetcode.cn/problems/daily-temperatures/
     * 思路: 单调栈,栈中元素为数组元素的index
     * 注: LinkedList可以实现队列,亦可以实现栈
     */
    public int[] dailyTemperatures(int[] temperatures) {
        int[] rns = new int[temperatures.length];
        Deque<Integer> stack = new LinkedList<Integer>();
        for(int i=0;i<temperatures.length;i++) {
            while(!stack.isEmpty() && temperatures[stack.peek()]< temperatures[i]) {
                //这种不断出栈的结果刚好满足单调栈的应用场景,所以可以使用栈的数据结构,避免堆排序的开销.
                int index = stack.pop();
                rns[index] = i-index;
            }
            stack.push(i);
        }
        return rns;
    }

    /**
     * 769. 最多能完成排序的块 https://leetcode.cn/problems/max-chunks-to-make-sorted/
     * 思路: 贪心
     * 一个符合要求的块需要满足,index 和 arr[index] 都在这个块中,从第一个元素开始找到当前块的边界即可.
     */
    public int maxChunksToSorted(int[] arr) {
        int currIndex = 0 , count = 0;
        int[] target = new int[arr.length];
        for(int i=0;i<arr.length;i++) {
            target[arr[i]] = i;
        }
        while(currIndex<arr.length) {
            currIndex = nextIndex(arr,target,currIndex)+1;
            count++;
        }
        return count;
    }

    /**
     * 根据当前块的起始index,找到当前块的最大的右边界index
     * @param arr 原始数组
     * @param target arr数组index和value的关系
     * @param currIndex 当前index
     * @return 最大的右边界index
     */
    private int nextIndex(int[] arr, int[] target, int currIndex) {
        int edgeIndex = Math.max(currIndex,target[currIndex]);
        for(int i=currIndex;i<=edgeIndex;i++) {
            int nextEdgeIndex = Math.max(i,target[i]);
            edgeIndex = Math.max(edgeIndex,nextEdgeIndex);
        }
        return edgeIndex;
    }

    /**
     * 785. 判断二分图 https://leetcode.cn/problems/is-graph-bipartite/
     * 思路: 并查集
     * 如果是二分图,则每组连接关系,index作为起点,所有终点都应该在另外一组,即graph每组元素都应该和index不在一组,
     * 如果发现index和graph[index]在一组, 则返回false
     */
    public boolean isBipartite(int[][] graph) {
        DisJointSet disJointSet = new DisJointSet(graph.length);
        for(int i=0;i<graph.length;i++) {
            int[] array = graph[i];
            for(int x: array) {
                if(disJointSet.isConnected(i,x)) return false;
                disJointSet.union(array[0],x);
            }
        }
        return true;
    }

    /**
     * 802. 找到最终的安全状态 https://leetcode.cn/problems/find-eventual-safe-states/
     * 思路: 拓扑排序
     * 从安全节点出发反向查找,能唯一遍历到(不存在成环可能性)的节点的集合就是结果
     */
    public List<Integer> eventualSafeNodes(int[][] graph) {
        int n = graph.length;
        HashSet<Integer>[] adTable = new HashSet[n];
        int[] inDegree = new int[n];
        List<Integer> rns = new ArrayList<Integer>();
        Queue<Integer> needVisitQueue = new LinkedList<>();
        for(int i=0;i<n;i++) {
            adTable[i] = new HashSet<>();
        }
        //这段逻辑体现倒置的有向边关系
        for(int i=0;i<n;i++) {
            for(int node: graph[i]) {
                adTable[node].add(i);
                inDegree[i]++;
            }
        }
        for(int i=0;i<n;i++) {
            if(inDegree[i] == 0) needVisitQueue.offer(i);
        }
        while(!needVisitQueue.isEmpty()) {
            int currentNode = needVisitQueue.poll();
            rns.add(currentNode);
            for(int node: adTable[currentNode]) {
                inDegree[node]--;
                if(inDegree[node] == 0) needVisitQueue.offer(node);
            }
        }
        //List排序简单写法
        Collections.sort(rns);
        return rns;
    }
}
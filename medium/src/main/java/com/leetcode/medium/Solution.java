package com.leetcode.medium;

import com.leetcode.base.ListNode;
import com.leetcode.base.TreeNode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
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
        reverse(nums,i+1,nums.length-1);
    }

    /**
     * 根据index两两交换元素
     */
    private void reverse(int[] nums, int i, int j) {
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


}

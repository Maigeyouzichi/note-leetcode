package com.leetcode.easy;

import java.util.LinkedList;
import java.util.Queue;

public class EyTreeNodeMaxDeep {

    private class TreeNode {
        private int value;
        private TreeNode left;
        private TreeNode right;

        public int getValue() {
            return value;
        }

        public void setValue(int value) {
            this.value = value;
        }

        public TreeNode getLeft() {
            return left;
        }

        public void setLeft(TreeNode left) {
            this.left = left;
        }

        public TreeNode getRight() {
            return right;
        }

        public void setRight(TreeNode right) {
            this.right = right;
        }
    }

    /**
     * 给定一个二叉树，找出其最大深度。
     * <p>
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     * <p>
     * 说明:叶子节点是指没有子节点的节点。
     * <p>
     * 示例：
     * 给定二叉树 [3,9,20,null,null,15,7]，
     * <p>
     *    3
     *   / \
     *  9  20
     *    /  \
     *   15   7
     * 返回它的最大深度 3 。
     * <p>
     * 请补全下边两种方式的不同实现代码：
     * <p>
     */

    /**
     * 数组结构
     */
    public static int getTreeDeepSize(Integer[] treeData) {
        if (treeData == null || treeData.length == 0) { return 0; }
        int depth = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);
        while (!queue.isEmpty()) {
            //当前这一层的处理的节点数
            int size = queue.size();
            while (size-- > 0) {
                int index = queue.poll();
                if (treeData[index] == null) {
                    //如果为null,当前节点就是null,就不用处理其子孩子
                    continue;
                }
                int leftIndex = 2 * index + 1;
                int rightIndex = 2 * index + 2;
                if (leftIndex < treeData.length) {
                    queue.offer(leftIndex);
                }
                if (rightIndex < treeData.length) {
                    queue.offer(rightIndex);
                }
            }
            depth++;
        }
        return depth;
    }

    /**
     * 二叉结构
     */
    public static int getTreeDeepSize(TreeNode treeData) {
        if(treeData == null) { return 0; }
        return Math.max(getTreeDeepSize(treeData.getLeft()),getTreeDeepSize(treeData.getRight())) + 1;
    }

    /**
     * 数组转二叉树
     */
    public static TreeNode buildTreeNode(Integer[] treeData) {
        if (treeData == null || treeData.length == 0 ) return null;
        Queue<EyTreeNodeMaxDeep.TreeNode> queue = new LinkedList<>();
        TreeNode rootNode = getTreeNodeInstance(treeData[0]);
        queue.offer(rootNode);
        int currentIndex = 1;
        while (!queue.isEmpty()) {
            TreeNode currentNode = queue.poll();

            if (currentIndex >= treeData.length) break;
            Integer leftValue = treeData[currentIndex++];
            if (leftValue != null) {
                TreeNode leftNode = getTreeNodeInstance(leftValue);
                currentNode.setLeft(leftNode);
                queue.offer(leftNode);
            }

            if (currentIndex >= treeData.length -1) break;
            Integer rightValue = treeData[currentIndex++];
            if (rightValue != null) {
                TreeNode rightNode = getTreeNodeInstance(rightValue);
                currentNode.setRight(rightNode);
                queue.offer(rightNode);
            }
        }
        return rootNode;
    }

    public static TreeNode getTreeNodeInstance(Integer value) {
        TreeNode treeNode = new EyTreeNodeMaxDeep().new TreeNode();
        treeNode.setValue(value);
        return treeNode;
    }

    public static void main(String[] args) {
        Integer[] treeData = {3,9,20,null,null,15,7};
        //测试数组形式
        System.out.println(getTreeDeepSize(treeData));
        //测试TreeNode形式
        TreeNode treeNode = buildTreeNode(treeData);
        System.out.println(getTreeDeepSize(treeNode));
    }

}
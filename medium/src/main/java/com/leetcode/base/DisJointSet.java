package com.leetcode.base;

/**
 * 并查集
 * @author lihao
 */
public class DisJointSet{

    /**index是当前节点,value是其父节点*/
    private final int[] arr;

    /**
     * 初始化数组
     */
    public DisJointSet(int n) {
        arr = new int[n];
        for(int i=0;i<n;i++) {
            arr[i] = i;
        }
    }

    /**
     * 合并两个节点: value变成共同的父节点
     */
    public void union(int x,int y) {
        int rootX = findRoot(x);
        int rootY = findRoot(y);
        if (rootX == rootY) {
            return;
        }
        arr[rootX] = rootY;
    }

    /**
     * 每次找到最上层的父节点: 即index == value
     */
    public int findRoot(int x) {
        return x==arr[x] ? x : (arr[x] = findRoot(arr[x]));
    }

    public int[] getArr() {
        return arr;
    }

    public boolean isConnected(int x, int y) {
        return findRoot(x) == findRoot(y);
    }

}
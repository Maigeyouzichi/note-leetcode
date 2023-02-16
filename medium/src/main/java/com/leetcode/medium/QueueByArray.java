package com.leetcode.medium;

/**
 * 数组实现队列
 * -- 线程不安全
 * @author lihao on 2023/2/16
 */
public class QueueByArray {

    int[] arr;
    int headIndex = 0;
    int tailIndex = 0;
    int size = 0;
    int capacity;

    public QueueByArray(int capacity) {
        arr = new int[capacity];
        this.capacity = capacity;
    }

    public synchronized boolean offer(int n) {
        if (size == arr.length) { return false; }
        arr[(tailIndex++)%capacity] = n;
        size++;
        return true;
    }

    public synchronized int poll() {
        if (size == 0) { return -1; }
        size--;
        return arr[(headIndex++)%capacity];
    }

    public int size() {
        return this.size;
    }

    public static void main(String[] args) {
        QueueByArray arr = new QueueByArray(5);
        arr.offer(1);
        System.out.println(arr.poll());

        arr.offer(2);
        arr.offer(3);
        System.out.println(arr.poll());
        System.out.println(arr.poll());

        arr.offer(2);
        arr.offer(3);
        System.out.println(arr.poll());
        System.out.println(arr.poll());

        arr.offer(2);
        arr.offer(3);
        System.out.println(arr.poll());
        System.out.println(arr.poll());

        arr.offer(2);
        arr.offer(3);
        System.out.println(arr.poll());
        System.out.println(arr.poll());

        arr.offer(2);
        arr.offer(3);
        System.out.println(arr.poll());
        System.out.println(arr.poll());

        arr.offer(2);
        arr.offer(3);
        System.out.println(arr.poll());
        System.out.println(arr.poll());

        arr.offer(2);
        arr.offer(3);
        arr.offer(4);
        arr.offer(5);
        arr.offer(6);
        arr.offer(7);
        System.out.println(arr.poll());
        System.out.println(arr.poll());
        System.out.println(arr.poll());
        System.out.println(arr.poll());
        System.out.println(arr.poll());
        System.out.println(arr.poll());
    }

}

package com.leetcode.easy;

import java.util.concurrent.ConcurrentHashMap;

/**
 * 安永面试 设计锁工具类
 * @author lihao on 2023/3/10
 */
public class EyLockUtil {
    private static final ConcurrentHashMap<String, Object> locks = new ConcurrentHashMap<>();

    /**
     * 同步执行锁
     * @param key  锁key
     * @param func 执行方法
     */
    public static void exec(String key, Runnable func) {
        Object lock = locks.getOrDefault(key, new Object());
        Object tmpLock = locks.putIfAbsent(key, lock);
        if (null != tmpLock) {
            lock = tmpLock;
        }
        synchronized (lock) {
            func.run();
        }
    }
}
package com.leetcode;

import java.util.concurrent.ConcurrentHashMap;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class EasyApplicationTests {

    @Test
    void contextLoads() {
        ConcurrentHashMap<String, String> map = new ConcurrentHashMap<>();
        System.out.println(map.putIfAbsent("a", "a"));
        System.out.println(map.putIfAbsent("a", "b"));
    }

}

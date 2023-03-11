package com.leetcode.easy;

import java.util.Arrays;

/**
 * 安永面试
 * 对一个字符串数组进行排序如下：（仔细观察输入输出数组）
 * 输入：arm=['D12','D12A',B,'CX','B1','D12B','C90B,'C100A','B0','C90A]
 * 输出：[B,'B0','B1','C90A,C90B,'C100A','CX',D12','D12A',D12B]
 * 请写出程序代码
 * @author lihao on 2023/3/10
 */
public class EyArraySort {
    public static void main(String[] args) {
        String[] arr = {"D12", "D12A", "B", "CX", "B1", "D12B", "C90B", "C100A", "B0", "C90A"};
        Arrays.sort(arr, (str1, str2) -> {
            int i = 0, j = 0;
            while (i < str1.length() && j < str2.length()) {
                char c1 = str1.charAt(i), c2 = str2.charAt(j);
                //数字部分比较
                if (Character.isDigit(c1) && Character.isDigit(c2)) {
                    int currentSum1 = 0, currentSum2 = 0;
                    while (i < str1.length() && Character.isDigit(str1.charAt(i))) {
                        currentSum1 = currentSum1 * 10 + (str1.charAt(i) - '0');
                        i++;
                    }
                    while (j < str2.length() && Character.isDigit(str2.charAt(j))) {
                        currentSum2 = currentSum2 * 10 + (str2.charAt(j) - '0');
                        j++;
                    }
                    if (currentSum1 != currentSum2) { return currentSum1 - currentSum2; }
                    //字母部分比较
                } else if (Character.isLetter(c1) && Character.isLetter(c2)) {
                    String currentStr1 = "", currentStr2 = "";
                    while (i < str1.length() && Character.isLetter(str1.charAt(i))) {
                        currentStr1 += str1.charAt(i);
                        i++;
                    }
                    while (j < str2.length() && Character.isLetter(str2.charAt(j))) {
                        currentStr2 += str2.charAt(j);
                        j++;
                    }
                    int currentCompareResult = currentStr1.compareTo(currentStr2);
                    if (currentCompareResult != 0) { return currentCompareResult; }
                }
            }
            return str1.length() - str2.length();
        });
        System.out.println(Arrays.toString(arr));
    }
}

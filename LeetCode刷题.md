# LeetCode刷题

### 1. 两数之和

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

**示例 1：**

```
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
```

**示例 2：**

```
输入：nums = [3,2,4], target = 6
输出：[1,2]
```

**示例 3：**

```
输入：nums = [3,3], target = 6
输出：[0,1]
```

链接：https://leetcode-cn.com/problems/two-sum


#### 解法一：暴力

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            //以i+1为起始搜索地址，避免返回结果里有重复元素
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] == target) {
                    return new int[]{i, j};
                }
            }
        }
        return null;    
    }
}
```

时间复杂度：两层 for 循环，O（n²）

空间复杂度：O（1）

#### 解法二：哈希表

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> table = new HashMap<>();
        int n = nums.length;
        for(int i = 0; i < n; i++) {
            //将所有元素与其对应的位置存入哈希表
            table.put(nums[i],i);
        }
        for (int i = 0; i < n; i++) {
            int toFind = target-nums[i];
            //如果哈希表中有我们需要寻找的数
            if(table.containsKey(toFind)) {
                //并且这个数并不是nums[i]本身
                int pos = table.get(toFind);
                if(pos != i) return new int[] {i,pos};
            }
        }
        return null;    
    }
}
```

时间复杂度：O（n），但是在leetcode上提交时，时间效率并不如暴力解法好

空间复杂度：O（n）

#### 解法三：哈希表优化

解法二用到的是两个for循环，可以只用一个for循环解决这道题。

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> table = new HashMap<>();
        int n = nums.length;
        for(int i = 0; i < n; i++) {
            if(table.containsKey(nums[i])) {
                //如果包含当前指针所在元素，那么说明已经找到，返回结果
                return new int[]{table.get(nums[i]),i};
            }
            //注意这里不是存放(nums[i],i)，而是(target-nums[i],i)
            //这样等循环遍历到了数组中某个等于target-nums[i]的数时
            //可以通过哈希表搜索直接获取到i
            table.put(target-nums[i],i);
        }
        return null;    
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

### 2. 两数相加

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。 

**示例 1：**

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```

**示例 2：**

```
输入：l1 = [0], l2 = [0]
输出：[0]
```

**示例 3：**

```
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```

链接：https://leetcode-cn.com/problems/add-two-numbers

思路：

两条链表同一位置元素相加，如有进位则将jinWei标记为1。

如此遍历直到两条链表尾且jinWei为0。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int jinWei = 0;
        ListNode head = null, p = null;
        while(l1 != null || l2 != null || jinWei != 0) {
            int num = 0;
            if(l1 != null) {
                num += l1.val;
                l1 = l1.next;
            }
            if(l2 != null) {
                num += l2.val;
                l2 = l2.next;
            }
            num += jinWei;
            if(head == null) {                
                head = new ListNode(num%10);
                p = head;
            } else{
                p.next = new ListNode(num%10);
                p = p.next;
            }            
            if(num >= 10) {
                jinWei = 1;
            }else {
                jinWei = 0;
            }
        } 
        return head;           
    }
}
```

### 3. 无重复字符的最长子串

给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。 

**示例 1:**

```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```


**示例 2:**

```
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```


**示例 3:**

```
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```


**示例 4:**

```
输入: s = ""输出: 0
```


链接：https://leetcode-cn.com/problems/longest-substring-without-repeating-characters

#### 解法一：哈希表

很容易想到用哈希表记录字符和其对应的位置。

```java
class Solution {    public int lengthOfLongestSubstring(String s) {        int ans = 0;        int start = 0; //从哪里开始截断重新开始计数        if(s.length() == 0) return ans;        HashMap<Character,Integer> hm = new HashMap<>();        for(int i = 0; i < s.length(); i++) {            char now = s.charAt(i);            if(hm.containsKey(now)) {                start = Math.max(start,hm.get(now)+1);            }            hm.put(now,i);            ans = Math.max(ans,i-start+1);        }        return ans;    }}
```

时间复杂度：O（n）

空间复杂度：使用了一个哈希表，判断子串中有没有重复的字符。由于 hm 中没有重复的字符，所以最长就是整个字符集，假设字符集的大小为 m ，那么 hm最长就是 m 。另一方面，如果字符串的长度小于 m ，是 n 。那么 hm 最长也就是 n 了。综上，空间复杂度为 O（min（m，n））。

#### 解法二：数组

哈希表的搜索方式本质上也是遍历搜寻，可以将哈希表优化为一个数组。字符的 ASCII 码值作为数组的下标，数组存储该字符所在字符串的位置。

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if(s.length() == 0) return 0;
        int ans = 0;
        int start = 0;
        int[] map = new int[128];
        char[] chars = s.toCharArray();//这里转换成char数组有利于提高速度
        
        for(int i = 0; i < chars.length; i++) {
            int cur = chars[i];
            if(map[cur] != 0) {
                start = Math.max(map[cur],start);      
            }
            map[cur] = i+1;
            ans = Math.max(ans, i-start+1);
        }
        return ans;
    }
}
```

时间复杂度：O（n）。

空间复杂度：O（1）。开辟的数组空间大小是不变的。

### 4. 寻找两个正序数组的中位数

给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

**示例 1：**

```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
```


**示例 2：**

```
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```


**示例 3：**

```
输入：nums1 = [0,0], nums2 = [0,0]
输出：0.00000
```


**示例 4：**

```
输入：nums1 = [], nums2 = [1]
输出：1.00000
```


**示例 5：**

```
输入：nums1 = [2], nums2 = []
输出：2.00000
```


链接：https://leetcode-cn.com/problems/median-of-two-sorted-arrays

#### 解法一：

因为是寻找中位数，所以可知实际上就是寻找第K小的数。

可以分别给两个数组两个指针，移动指向比较小的那个数的指针

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        if((m+n)%2 == 0) return ((double)getK(nums1,nums2,(m+n+1)/2) + (double)getK(nums1, nums2, (m+n+2)/2))/2;        
        return getK(nums1,nums2,(m+n+1)/2);
    }
    private int getK(int[] nums1, int[] nums2, int k) {
        int p1 = 0, p2 = 0;
        int m = nums1.length, n = nums2.length;
        while(p1 < m || p2 < n) {
            if(p1+p2+1 == k) {
                if(p1 == m) {//如果第一个指针已经越界
                    return nums2[p2];//直接返回数组二里的数
                }else if(p2 == n) {//如果第二个指针越界
                    return nums1[p1];//返回数组一中的数
                }else {
                    //如果两个指针都没越界，那么返回较小的那个
                    return Math.min(nums1[p1],nums2[p2]);
                }
            }

            if(p1 == m) {//如果第一个数组已经遍历完
                p2++;//移动数组二的指针
            }else if(p2 == n){//如果第二个数组已经遍历完
                p1++;//移动数组一的指针
            }else {//如果两个数组都还没遍历完
                //比较大小
                if(nums1[p1] < nums2[p2]) {
                    p1++;
                }else {
                    p2++;
                }
            }
        }
        return -1;
    }
}
```

时间复杂度：O（m+n），达不到题目中要求的O（log（m+n））。

空间复杂度： O（1）。

#### 解法二：二分法

题目要求的时间复杂度是O（log（m+n）），很自然会联想到二分查找法。其实就是在解法一的基础上运用了二分查找法来筛查到第K小的数。

```java
class Solution {    public double findMedianSortedArrays(int[] nums1, int[] nums2) {        int m = nums1.length, n = nums2.length;        if((m+n)%2 == 1) return getK(nums1,0,nums2,0,(m+n+1)/2);        return ((double)getK(nums1,0,nums2,0,(m+n+1)/2) + (double)getK(nums1,0,nums2,0,(m+n+2)/2))/2;    }    private int getK(int[] nums1, int start1, int[] nums2, int start2, int k) {        if(start1 == nums1.length) return nums2[start2 + k - 1];        if(start2 == nums2.length) return nums1[start1 + k - 1];        if(k == 1) return nums1[start1] <= nums2[start2] ? nums1[start1] : nums2[start2];        //防止数组越界        int len1 = Math.min(start1 + k/2 - 1, nums1.length - 1);        int len2 = Math.min(start2 + k/2 - 1, nums2.length - 1);        if(nums1[len1] <= nums2[len2]) {            //舍弃掉nums1[len1]及之前的所有数，从len1+1位置开始搜寻            //剩下的所有数中要找寻第k-(len1-start1+1)小的数            return getK(nums1, len1 + 1, nums2, start2, k-(len1-start1+1));        }        return getK(nums1, start1, nums2, len2 + 1, k-(len2-start2+1));    }}
```

### 5. 最长回文子串

给你一个字符串 s，找到 s 中最长的回文子串。

**示例 1：**

```
输入：s = "babad"输出："bab"解释："aba" 同样是符合题意的答案。
```


**示例 2：**

```
输入：s = "cbbd"输出："bb"
```

**示例 3：**

```
输入：s = "a"输出："a"
```


**示例 4：**

```
输入：s = "ac"输出："a"
```


链接：https://leetcode-cn.com/problems/longest-palindromic-substring

#### 解法一：暴力解

暴力求解，列举所有的子串，判断是否为回文串。

```java
class Solution {    public String longestPalindrome(String s) {        if(s.length() == 1) return s;        char[] array = s.toCharArray();        int longest = 1;//因为每单个字符一定是回文串，所以可以确定的是最长的回文串一定大于等于1        int left = 0, right = 0;        for(int i = 0; i < array.length - longest; i++) {            //一点简单的剪枝操作            for(int j = i+longest; j < array.length; j++) {                if(isPalindrome(array, i, j)) {                    longest = j - i + 1;                    left = i;                    right = j;                }            }        }        return s.substring(left,right+1);    }    private boolean isPalindrome(char[] array, int left, int right) {        while(left < right) {            if(array[left] == array[right]) {                left++;                right--;            }else {                return false;            }        }        return true;    }}
```

时间复杂度：两层 for 循环 O（n²），for 循环里边判断是否为回文，O（n），所以时间复杂度为 O（n³）。

空间复杂度：O（n），如果不将字符串转化成char数组则只需要O（1）。

#### 解法二：暴力优化（动态规划）

用一个二维布尔数组isP[i] [j] 表示从位置i到j是否为回文串，如果i位置和j位置的字符相同，且isP[i+1] [j-1]为真，那么可以得出isP[i] [j]为真。

```java
class Solution{    public String longestPalindrome(String s) {        if(s.length() == 1) return s;        int length = s.length();        boolean[][] isP = new boolean[length][length];        char[] chars = s.toCharArray();        int left = 0, right = 0;        for(int i = 0; i < length; i++) {            isP[i][i] = true;        }        for(int len = 2; len <= length; len++) {            for(int start = 0; start <= length - len; start++){                int end = start + len - 1;                isP[start][end] = (chars[start] == chars[end]) && (len == 2 || isP[start+1][end-1]);                if(isP[start][end]) {                    left = start;                    right = end;                }            }        }        return s.substring(left,right+1);    }}
```

时间复杂度：两层循环，O（n²）。

空间复杂度：用二维数组保存每个子串的情况，O（n²）。

这里的空间复杂度还可以优化，可以尝试用一维数组来表示isP[j]表示。由于isP[i] [j]是从isP[i+1] [j-1]得到的，所以知道以i为起始的isP必须要先得到i+1的isP。遍历顺序应当是倒序。

```java
class Solution{    public String longestPalindrome(String s) {        if(s.length() == 1) return s;        int length = s.length();        boolean[] isP = new boolean[length];        char[] chars = s.toCharArray();        int left = 0, right = 0, longest = 1;        for(int i = length - 1; i >= 0; i--) {            for(int j = length - 1; j >= i; j--){                isP[j] = (chars[i] == chars[j]) && (j - i <= 2 || isP[j-1]);                if(isP[j] && j - i + 1 > longest) {                    left = i;                    right = j;                    longest = j - i + 1;                }            }        }        return s.substring(left,right+1);    }}
```

时间复杂度：两层循环，O（n²）。

空间复杂度：O（n）。

#### 解法三：中心扩展

以中间为搜寻起点，如果是回文子串则向两边散开。

```java
class Solution {    public String longestPalindrome(String s) {        if(s.length() == 1) return s;        char[] array = s.toCharArray();        int[] memo = {0,0,0};//保存最长回文长度，左边界索引，右边界索引        for(int i = 1; i < array.length - memo[0]/2; i++) {            countPalindrome(array, i, i, memo);            countPalindrome(array, i-1, i, memo);        }        return s.substring(memo[1],memo[2]+1);    }    private void countPalindrome(char[] a, int left, int right, int[] memo) {        while(left >= 0 && right < a.length && a[left] == a[right]) {            left--;            right++;        }        if(right - left - 1 > memo[0]) {            memo[0] = right - left - 1;            memo[1] = left + 1;            memo[2] = right - 1;        }    }}
```

时间复杂度：一层 for 循环 O（n），for 循环里边判断是否为回文，O（n），所以时间复杂度为 O（n²）。

空间复杂度：O（n），如果不将字符串转化成char数组则只需要O（1）。

#### 解法四：动态规划

如果一个字符串是回文串，那么其正序和倒序读起来都是一样的，以字符串“caba“为例，将其倒过来是"abac"，最长公共子串是"aba"，所以"aba"就是我们要寻找的最长回文子串。

把两个字符串分别以行和列组成一个二维矩阵，如果字符串1的i位置和字符串2的j位置相等，那么就用dp[i] [j]=1 +dp[i-1] [j-1]更新。

```java
class Solution {    public String longestPalindrome(String s) {        if(s.length() == 1) return s;        char[] array = s.toCharArray();        int longest = 1;//因为每单个字符一定是回文串，所以可以确定的是最长的回文串一定大于等于1        int left = 0, right = 0;        int[][] dp = new int[array.length][array.length];        //以逆序数组为行，正序数组为列        for(int i = 0; i < array.length; i++) {            for(int j = 0; j < array.length; j++) {                if(array[j] == array[array.length - 1 - i]) {                    dp[i][j] = ((i > 0 && j > 0) ? dp[i-1][j-1] : 0) + 1;                    if(dp[i][j] > longest) {                        if(j - dp[i][j] + 1 == array.length - 1 - i) {                            //确认一下下标无误，即正序开始时的下标等于array.length - 1减去逆序位置i                            //如果不加这一步确认那么遇到例如"aacabdkacaa"时，会出现dp[3][3]最大，left=7，right = 3                            longest = dp[i][j];                            right = j;                            left = array.length - 1 - i;                        }                                            }                }            }        }        return s.substring(left,right+1);    }}
```

时间复杂度：两层循环，O（n²）。但是不如中心扩展效率好，因为中心扩展法有剪枝操作，而利用动态规划无法这么做。

空间复杂度：一个二维数组，O（n²）。

可以只用一维数组来优化空间复杂度。

```java
class Solution {    public String longestPalindrome(String s) {        int length = s.length();        if(length == 1) return s;                char[] array = s.toCharArray();        int longest = 1;//因为每单个字符一定是回文串，所以可以确定的是最长的回文串一定大于等于1        int left = 0, right = 0;        int[] dp = new int[length];        //以逆序数组为行，正序数组为列        for(int i = 0; i < length; i++) {            for(int j = length - 1; j >= 0; j--) {                if(array[j] == array[array.length - 1 - i]) {                    dp[j] = ((j > 0) ? dp[j-1] : 0) + 1;                    if(dp[j] > longest) {                        if(j - dp[j] + 1 == array.length - 1 - i) {                            //确认一下下标无误，即正序开始时的下标等于array.length - 1减去逆序位置                            longest = dp[j];                            right = j;                            left = array.length - 1 - i;                        }                                            }                }else {                    //之前二维数组，每次用的是不同的列，所以不用置 0 。                    dp[j] = 0;                }            }        }        return s.substring(left,right+1);    }}
```

### 6. Z 字形变换

将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：

P    A    H    N
A P L S  I  I  G
Y     I     R
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。

请你实现这个将字符串进行指定行数变换的函数：

string convert(string s, int numRows);

**示例 1：**

```
输入：s = "PAYPALISHIRING", numRows = 3
输出："PAHNAPLSIIGYIR"
```

**示例 2：**

```
输入：s = "PAYPALISHIRING", numRows = 4
输出："PINALSIGYAHRPI"
解释：
P     I    N
A   L S  I G
Y A   H R
P     I
```

**示例 3：**

```
输入：s = "A", numRows = 1
输出："A"
```

链接：https://leetcode-cn.com/problems/zigzag-conversion

#### 解法一：找规律，求通项

当输入numRows = 5时：

可以看到第0行间隔是8（（numRows - 1）*2）

第一行间隔是6（（numRows - 1 - 1）*2）和2（1 * 2）

第二行间隔是4（（numRows - 1 - 2）*2）和4（2 * 2）

第三行间隔是2（（numRows - 1 - 3）*2）和6（3 * 2）

第四行间隔是8（4 * 2）

![](http://windliang.oss-cn-beijing.aliyuncs.com/6_3.jpg)

```java
class Solution {    public String convert(String s, int numRows) {        if(numRows == 1 || s.length() <= numRows) return s;        char[] chars = s.toCharArray();        int row = 0;        StringBuilder sb = new StringBuilder();        while(row < numRows) {            int pos = row;            //当step1和step2为0时要特殊处理            int step1 = (numRows - row - 1) * 2 == 0 ? row * 2 : (numRows - row - 1) * 2;            int step2 = row * 2 == 0 ? step1 : row * 2;            boolean isS1 = true;            while(pos < chars.length) {                sb.append(chars[pos]);                if(isS1){                    pos += step1;                    isS1 = false;                }else {                    pos += step2;                    isS1 = true;                }            }            row++;        }        return sb.toString();    }}
```

#### 解法二：Z字存储

按照写 Z 的过程，遍历每个字符，然后将字符存到对应的行中。用upNotDown表示当前进行方向，如果为false代表要往下写，所以row要加一；如果为true，row要减一。

```java
class Solution {    public String convert(String s, int numRows) {        if(numRows == 1 || s.length() <= numRows) return s;        char[] chars = s.toCharArray();        List<StringBuilder> rows = new ArrayList<StringBuilder>();        boolean upNotDown = false;        int row = 0;        for(int i = 0; i < numRows; i++) {            rows.add(new StringBuilder());        }        for(char c : chars) {            rows.get(row).append(c);                        if(upNotDown) {                row--;                if(row == 0) upNotDown = false;            }else {                row++;                if(row == numRows - 1) upNotDown = true;            }        }        StringBuilder sb = new StringBuilder();        for (StringBuilder r : rows) sb.append(r);                return sb.toString();    }}
```

### 7.  整数反转

给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。

假设环境不允许存储 64 位整数（有符号或无符号）。

**示例 1：**

```
输入：x = 123
输出：321
```

**示例 2：**

```
输入：x = -123
输出：-321
```


**示例 3：**

```
输入：x = 120
输出：21
```

链接：https://leetcode-cn.com/problems/reverse-integer



本题唯一的难点就在于溢出，例如若输入为1534236469，倒置过来不应该是 9646324351 吗。其实题目里讲了，int 的范围是 
$$
[-2^{31} ,2^{31}-1]
$$
也就是 [ -2147483648 , 2147483647 ] 。明显 9646324351 超出了范围，造成了溢出。所以我们需要在输出前，判断是否溢出。可以直接将保存数据类型int改为long，返回数值时直接利用强转int即可。

```java
class Solution {
    public int reverse(int x) {
        //个位数单独处理，直接返回即可
        if((x >= 0 && x <= 9) || (x < 0 && x > -10)) return x;
        boolean negative = false;
        if(x < 0) {
            negative = true;
            x = Math.abs(x);
        }
        long res = 0;
        while(x > 0) {
            res = res * 10 + x % 10;
            if (res > Integer.MAX_VALUE || res < Integer.MIN_VALUE) return 0;
            x /= 10;
        }
        if(negative) res = - res;
        return (int)res;
    }
}
```

### 8. 字符串转换整数 (atoi)

请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。

函数 myAtoi(string s) 的算法如下：

读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
返回整数作为最终结果。

注意：

本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。

**示例 1：**

```
输入：s = "42"
输出：42
解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
第 1 步："42"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："42"（读入 "42"）
           ^
解析得到整数 42 。
由于 "42" 在范围 [-231, 231 - 1] 内，最终结果为 42 。
```

**示例 2：**

```
输入：s = "   -42"
输出：-42
解释：
第 1 步："   -42"（读入前导空格，但忽视掉）
            ^
第 2 步："   -42"（读入 '-' 字符，所以结果应该是负数）
             ^
第 3 步："   -42"（读入 "42"）
               ^
解析得到整数 -42 。
由于 "-42" 在范围 [-231, 231 - 1] 内，最终结果为 -42 。
```


**示例 3：**

```
输入：s = "4193 with words"
输出：4193
解释：
第 1 步："4193 with words"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："4193 with words"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："4193 with words"（读入 "4193"；由于下一个字符不是一个数字，所以读入停止）
             ^
解析得到整数 4193 。
由于 "4193" 在范围 [-231, 231 - 1] 内，最终结果为 4193 。
```


**示例 4：**

```
输入：s = "words and 987"
输出：0
解释：
第 1 步："words and 987"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："words and 987"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："words and 987"（由于当前字符 'w' 不是一个数字，所以读入停止）
         ^
解析得到整数 0 ，因为没有读入任何数字。
由于 0 在范围 [-231, 231 - 1] 内，最终结果为 0 。
```


**示例 5：**

```
输入：s = "-91283472332"
输出：-2147483648
解释：
第 1 步："-91283472332"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："-91283472332"（读入 '-' 字符，所以结果应该是负数）
          ^
第 3 步："-91283472332"（读入 "91283472332"）
                     ^
解析得到整数 -91283472332 。
由于 -91283472332 小于范围 [-231, 231 - 1] 的下界，最终结果被截断为 -231 = -2147483648 。
```

链接：https://leetcode-cn.com/problems/string-to-integer-atoi



本题要尤其注意测试样例 “  +4” 、“+-12”、"00000-42a1234"。这代表除了‘-’，'+‘也要考虑，且一旦前面出现了有效的字符后面再出现除数字以外的任何字符都是不作数的。

```java
class Solution {    public int myAtoi(String s) {        if(s.length() == 0) return 0;        char[] chars = s.toCharArray();        long res = 0; //结果用long防止溢出        boolean negative = false; //表示是否为负数        boolean hasMet = false; //表示之前已经遇到了有效字符        for(char c : chars) {            if(!hasMet && c == '-' && res == 0) {                negative = true;                hasMet = true;            }else if(!hasMet && c == '+' && res == 0){                negative = false;                hasMet = true;            }else if(!hasMet && c == ' ' && res == 0){                continue;            }else{                            int cur = c - 48;                if(cur >= 0 && cur <= 9){                     hasMet = true;                                       res = res * 10 + cur;                }else {                    break;                }                if(negative && res > 2147483648L) return -2147483648;                if(!negative && res >= 2147483648L) return 2147483647;            }        }        if(negative) res = -res;                return (int)res;    }}
```



### 9. 回文数

给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。例如，121 是回文，而 123 不是。



**示例 1：**

```
输入：x = 121
输出：true
```


**示例 2：**

```
输入：x = -121
输出：false
解释：从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
```


**示例 3：**

```
输入：x = 10
输出：false
解释：从右向左读, 为 01 。因此它不是一个回文数。
```


**示例 4：**

```
输入：x = -101
输出：false
```

链接：https://leetcode-cn.com/problems/palindrome-number、



#### 解法一：整数反转

题目要求了不能用将int转成string的方法来写这题，那么可以用第七题整数反转的方法。

```java
class Solution {
    public boolean isPalindrome(int x) {
        if(x >= 0 && x < 10) return true; //只有个位数一定是回文
        if(x < 0 || x%10 == 0) return false; //负数或者是末尾为0的情况都排除
        int reverse = 0, tmp = x;
        while(tmp > 0) {
            reverse = reverse * 10 + tmp % 10;
            tmp /= 10;
        }
        return (x == reverse);
    }
}

```

时间复杂度：和求转置一样，x 有多少位，就循环多少次，所以是 O（log（x）） 。

空间复杂度：O（1）。

#### 解法二：整数反转一半

其实，我们只需要将右半部分倒置然后和左半部比较就可以了。

```java
class Solution {
    public boolean isPalindrome(int x) {
        if(x >= 0 && x < 10) return true; //只有个位数一定是回文
        if(x < 0 || x%10 == 0) return false; //负数或者是末尾为0的情况都排除
        
        int right_reverse = 0;
        int left = x;
        while(left > right_reverse) {            
            right_reverse = right_reverse*10 + left%10;//右侧翻转
            left = left/10;            
        }
        return left == right_reverse || left == right_reverse/10;
    }
}
```

时间复杂度： O（log（x）） 

空间复杂度：O（1）

### 10. 正则表达式匹配*

给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

**示例 1：**

```
输入：s = "aa" p = "a"
输出：false
解释："a" 无法匹配 "aa" 整个字符串。
```


**示例 2:**

```
输入：s = "aa" p = "a*"
输出：true
解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```


**示例 3：**

```
输入：s = "ab" p = ".*"
输出：true
解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
```


**示例 4：**

```
输入：s = "aab" p = "c*a*b"
输出：true
解释：因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
```


示例 5：

```
输入：s = "mississippi" p = "mis*is*p*."
输出：false
```

链接：https://leetcode-cn.com/problems/regular-expression-matching



#### 解法一：递归

```java
class Solution {
    public boolean isMatch(String s, String p) {
        //转char数组是为了加快速度
        char[] as = s.toCharArray();
        char[] ap = p.toCharArray();
        return match(as,0,ap,0);
    }
    private boolean match(char[] as, int start1, char[] ap, int start2) {
        //这里不用if(as.length - start1 == 0) return ap.length - start2 == 0
        //的原因是，"*"可以匹配零个或者多个，所以p的长度可以长于s也可以短于s，
        //但是当s到达末尾时，p必须到达末尾。反之，若p到达末尾，s还可以有若干个相同字符待匹配。
        if(ap.length - start2 == 0) return as.length - start1 == 0;

        boolean firstMatch = (as.length - start1 != 0 && (as[start1] == ap[start2] || ap[start2] == '.'));

        if(ap.length - start2 >= 2 && ap[start2+1] == '*') {
            //两种情况
            //1. pattern 直接跳过两个字符。表示 * 前边的字符出现 0 次
            //2. pattern 不变，例如 text = aa ，pattern = a*，第一个 a 匹配，
            //然后 text 的第二个 a 接着和 pattern 的第一个 a 进行匹配。表示 * 用前一个字符替代。
            return match(as, start1, ap, start2+2) || (firstMatch && match(as, start1+1, ap, start2));
        }else {
            return firstMatch && match(as, start1+1, ap, start2+1);
        }
    }
}
```

#### 解法二：动态规划

[按照题解来的](https://leetcode-cn.com/problems/regular-expression-matching/solution/shou-hui-tu-jie-wo-tai-nan-liao-by-hyj8/)

一个常用的trick就是申请多一维的空间，例如对于长度为m和n的字符串，dp数组就要申请[m+1] [n+1]，这样会避免掉很多麻烦。

```java
class Solution {
    public boolean isMatch(String s, String p) {
        if(p.length() == 0) return s.length() == 0;
        //转char数组是为了加快速度
        char[] as = s.toCharArray();
        char[] ap = p.toCharArray();
        //dp[i][j]表示s的前i位置与p的前j位置相匹配
        boolean[][] dp = new boolean[as.length+1][ap.length+1];
        //字符串都是空必然匹配得上
        dp[0][0] = true;
        //如果p为空，s不为空必然匹配不上
        //如果s为空，p不为空
        for(int i = 1; i <= ap.length; i++) {
           if(ap[i-1] == '*') {
               dp[0][i] = dp[0][i-2];
           }
        }        
        for(int i = 1; i <= as.length; i++) {
            for(int j = 1; j <= ap.length; j++) {
                if(ap[j-1] == as[i-1] || ap[j-1] == '.') {
                    //当前字符匹配上了或者是p当前字符是'.'
                    dp[i][j] = dp[i-1][j-1];
                }else if(ap[j-1] == '*') {
                    if(ap[j-2] == as[i-1] || ap[j-2] == '.') {//这里尤其需要注意很容易出错
                        //如果p在*之前的那个字符和s的当前字符相同
                        //那么可能是该字符出现0次或1次及以上
                        
                        //实际上dp[i-1][j-2]这一项可以不要，
                        //因为在计算dp[i-1][j]的时候是用dp[i-1][j-2]||dp[i-2][j]包含这一项在内了
                        dp[i][j] = dp[i][j - 2] || dp[i - 1][j - 2] || dp[i - 1][j];
                    }else {
                        //如果p在*之前的那个字符和s的当前字符不同
                        //那么只可能是该字符出现0次
                        dp[i][j] = dp[i][j-2];
                    }
                }
            }
        }
        return dp[as.length][ap.length];
    }
}
```

时间复杂度：O（mn）。

空间复杂度：O（mn）。

上面的方法是从前往后遍历，也可以从后往前遍历。就是当前位置要根据之后位置的状态来决定。其实可以看成和递归是一个道理，代码也很像递归的代码。

```java
class Solution {
    public boolean isMatch(String s, String p) {
        if(p.length() == 0) return s.length() == 0;
        //转char数组是为了加快速度
        char[] as = s.toCharArray();
        char[] ap = p.toCharArray();
        //dp[i][j]表示s的前i位置与p的前j位置相匹配
        boolean[][] dp = new boolean[as.length+1][ap.length+1];
        //从后往前遍历时，字符串尾部的位置表示字符串都是空，必然匹配得上
        dp[as.length][ap.length] = true;
        
        for(int i = as.length; i >= 0; i--) {
            for(int j = ap.length; j >= 0; j--) {
                if(i == as.length && j == ap.length) continue;
                boolean firstMatch = (i != as.length && j != ap.length && (as[i] == ap[j] || ap[j] == '.'));
                if(j + 1 < ap.length && ap[j+1] == '*') {
                    dp[i][j] = dp[i][j+2] || (firstMatch && dp[i+1][j]);
                }else {
                    dp[i][j] = firstMatch && dp[i+1][j+1];
                }
            }
        }
        return dp[0][0];
    }
}
```

时间复杂度：O（mn）。

空间复杂度：O（mn）。

本题的动态规划方法是没有办法优化成一维数组的，但是可以把二维数组优化得小一点。懒得写了。

### 11. 盛最多水的容器

给你 n 个非负整数 a~1~，a~2~，...，a~n~，每个数代表坐标中的一个点 (i, a~i~) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, a~i~) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器。

**示例 1：**

<img src="https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg" style="zoom:67%;" />

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

**示例 2：**

```
输入：height = [1,1]
输出：1
```

**示例 3：**

```
输入：height = [4,3,2,1,4]
输出：16
```

**示例 4：**

```
输入：height = [1,2,1]
输出：2
```


链接：https://leetcode-cn.com/problems/container-with-most-water

#### 解法一：暴力解法

遍历所有的柱子组合，求出最大的。

```java
class Solution {
    public int maxArea(int[] height) {
        int vel = 0;
        for(int i = 0; i < height.length; i++) {
            for(int j = i+1; j < height.length; j++) {
                vel = Math.max(vel,Math.min(height[i],height[j])*(j-i));
            }
        }
        return vel;
    }
}
```

时间复杂度：O（n^2^）超出时间限制

空间复杂度：O（1）

#### 解法二：双指针

用双指针一个从头开始往后移动，一个从尾部开始往前移动。

为了使容量大，要么是容器很宽，要么是容器很高，那么为了弥补容器底变小，就需要让高度变高，所以移动高度较为矮的那个指针。

```java
class Solution {
    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1, vel = 0;
        while(l < r) {
            vel = Math.max((r - l) * Math.min(height[l],height[r]), vel);
            if(height[l] < height[r]) {
                l++;
            }else {
                r--;
            }
        }
        return vel;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 12. 整数转罗马数字

罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

| 字符 | 数值 |
| ---- | ---- |
| I    | 1    |
| V    | 5    |
| X    | 10   |
| L    | 50   |
| C    | 100  |
| D    | 500  |
| M    | 1000 |

例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

- I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
- X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
- C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。

给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。

**示例 1:**

```
输入: 3
输出: "III"
```

**示例 2:**

```
输入: 4
输出: "IV"
```

**示例 3:**

```
输入: 9
输出: "IX"
```

**示例 4:**

```
输入: 58
输出: "LVIII"
解释: L = 50, V = 5, III = 3.
```

**示例 5:**

```
输入: 1994
输出: "MCMXCIV"
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

链接：https://leetcode-cn.com/problems/integer-to-roman

把所有的表示单元（1、5、9等）列在数组中，将4、9、40、90...这些数也作为表示单元，存入数组。

```java
class Solution {
    public String intToRoman(int num) {
        int yuShu = 0;
        int[] nums = {1,4,5,9,10,40,50,90,100,400,500,900,1000};
        String[] s = {"I","IV","V","IX","X","XL","L","XC","C","CD","D","CM","M"};
        StringBuilder res = new StringBuilder();
        for(int i = 12; i >= 0; i--) {
            yuShu = num / nums[i];
            num = num % nums[i]; 
            while(yuShu-- > 0) {
                res.append(s[i]);
            }
        }
        return res.toString();
    }
}
```

时间复杂度：O（1）

空间复杂度：O（1）

这题感觉还是比较简单的，很快就能有思路。

### 13. 罗马数字转整数

罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

| 字符 | 数值 |
| ---- | ---- |
| I    | 1    |
| V    | 5    |
| X    | 10   |
| L    | 50   |
| C    | 100  |
| D    | 500  |
| M    | 1000 |

例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

- I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
- X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
- C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。

给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

 

**示例 1:**

```
输入: "III"
输出: 3
```

**示例 2:**

```
输入: "IV"
输出: 4
```

**示例 3:**

```
输入: "IX"
输出: 9
```

**示例 4:**

```
输入: "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.
```

**示例 5:**

```
输入: "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

链接：https://leetcode-cn.com/problems/roman-to-integer

#### 解法一

与上一题不同的是，这里没有将4、9等特殊情况也存入，所有的对应字母都是以char型数组存储的。当遇到'I'、'X'、'C'时要判断下一位字母是否会与当前字幕构成特殊情况组合。

```java
class Solution {
    public int romanToInt(String s) {
        int[] nums = {1,5,10,50,100,500,1000};
        char[] map = {'I','V','X','L','C','D','M'};
        int res = 0, index = map.length - 1;
        char[] chars = s.toCharArray();
        for(int i = 0; i < chars.length; i++) {
            int cur = 0;
            while(index > 0 && chars[i] != map[index]) {
                index--;
            }
            cur = nums[index];
            if(i < chars.length - 1 && (index == 0 || index == 2 || index == 4)) {
                if(chars[i+1] == map[index+1]) {
                    i++;
                    cur = nums[index+1] - cur;
                }else if(chars[i+1] == map[index+2]) {
                    i++;
                    cur = nums[index+2] - cur;
                }
            }
            res += cur;
        }
        return res;
    }
}
```

时间复杂度：O（n），n为字符串s的长度

空间复杂度：O（1）

#### 解法二

提前减去两倍该减的部分。[按照题解来](https://leetcode.com/problems/roman-to-integer/description/)

```java
class Solution {
    public int romanToInt(String s) {
        int sum=0;
        if(s.indexOf("IV")!=-1){sum-=2;}
        if(s.indexOf("IX")!=-1){sum-=2;}
        if(s.indexOf("XL")!=-1){sum-=20;}
        if(s.indexOf("XC")!=-1){sum-=20;}
        if(s.indexOf("CD")!=-1){sum-=200;}
        if(s.indexOf("CM")!=-1){sum-=200;}

        char c[]=s.toCharArray();
        int count=0;
        for(;count<=s.length()-1;count++){
            if(c[count]=='M') sum+=1000;
            if(c[count]=='D') sum+=500;
            if(c[count]=='C') sum+=100;
            if(c[count]=='L') sum+=50;
            if(c[count]=='X') sum+=10;
            if(c[count]=='V') sum+=5;
            if(c[count]=='I') sum+=1;

        }
        return sum;
    }
}
```

时间复杂度：O（n），n为字符串s的长度

空间复杂度：O（1）

#### 解法三

记录下前一个值与当前值，若当前值大于前一个值，那么代表sum要减去前一个值，否则是加。

```java
class Solution {
    public int romanToInt(String s) {
        char[] c=s.toCharArray();
        int pre = getVal(c[0]);
        int sum = 0;
        int cur = 0;
        for(int i = 1; i < c.length; i++){
            cur = getVal(c[i]);
            if(pre < cur) {
                sum -= pre;
            }else {
                sum += pre;
            }
            pre = cur;
        }
        sum += pre;
        return sum;
    }
    private int getVal(char c) {
        switch(c) {
            case 'M' : return 1000;
            case 'D' : return 500;
            case 'C' : return 100;
            case 'L' : return 50;
            case 'X' : return 10;
            case 'V' : return 5;
            case 'I' : return 1;
            default : return -1;
        }
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 14. 最长公共前缀

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。

 

**示例 1：**

```
输入：strs = ["flower","flow","flight"]
输出："fl"
```

**示例 2：**

```
输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。
```

链接：https://leetcode-cn.com/problems/longest-common-prefix

#### 解法一：纵向扫描

从下标0开始，判断每一个字符串的下标0，判断是否全部相同。直到遇到不全部相同的下标，或是下标超出了字符串的边界。

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if(strs.length == 0) return "";
        if(strs.length == 1) return strs[0];
        StringBuilder path = new StringBuilder();
        int[] map = new int[26];
        int index = 0;
        boolean end = false;//标志位，是否结束while循环
        while(!end) {
            for(int i = 0; i < strs.length; i++) {
                //如果index超过了当前的字符串长度那么必然要停止while循环
                if(index >= strs[i].length()) {
                    end = true;
                    break;
                }
                //如果当前字符串的index字母数对应不上也要停止循环
                if(++map[strs[i].charAt(index)-'a'] != i+1) {
                    end = true;
                    break;
                }
            }
            if(!end) {
                path.append(strs[0].charAt(index));
                map[strs[0].charAt(index)-'a'] = 0;//对index字母数量复位0
                index++;
            }            
        }
        return path.toString();
    }
}
```

时间复杂度：O（n*m）

空间复杂度：O（m），主要是path大小会随着给的输入不同而变化

其实这里也并不一定需要用数组来存储对应关系，有些空间浪费了，其实只需要用一个char变量来保存前一个字符串index对应的字母即可。作了修改后如下。

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if(strs.length == 0) return "";
        if(strs.length == 1) return strs[0];
        StringBuilder path = new StringBuilder();
        char prev = ' ';
        int index = 0;
        boolean end = false;//标志位，是否结束while循环
        while(!end) {
            for(int i = 0; i < strs.length; i++) {
                //如果index超过了当前的字符串长度那么必然要停止while循环
                if(index >= strs[i].length()) {
                    end = true;
                    break;
                }
                if(i == 0) {
                    prev = strs[0].charAt(index);
                }else if(prev != strs[i].charAt(index)) {//如果当前字符串index指向的字母与prev不同
                    end = true;
                    break;
                }
            }
            if(!end) {
                path.append(strs[0].charAt(index));
                index++;
            }            
        }
        return path.toString();
    }
}
```

时间复杂度：O（n*m）

空间复杂度：O（m）

#### 解法二：横向扫描

任选一个字符串作为公共前缀，然后到下一个字符串判断是否以该字符串为开头，若不是，就删减到符合要求为止。用到了String里自带的startsWith()方法，其实有点作弊的意思了。要是自己写也是可以的，但肯定没有库函数来的速度快。

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if(strs.length==0)return "";
        //先任选一个作为公共前缀
        String s=strs[0];
        for (String string : strs) {
            while(!string.startsWith(s)){
                if(s.length()==0)return "";
                //公共前缀不匹配就删减一位
                s=s.substring(0,s.length()-1);
            }
        }
        return s;
    }
}
```

时间复杂度：O（n*m）

空间复杂度：O（m）

### 15. 三数之和

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

 **示例 1：**

```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
```

**示例 2：**

```
输入：nums = []
输出：[]
```

**示例 3：**

```
输入：nums = [0]
输出：[]
```

链接：https://leetcode-cn.com/problems/3sum

#### 解法一：哈希表

这题如果用暴力解法必然是会超时的，想着用哈希表来存储两个数的和与他们的最大坐标。

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums.length < 3) return res;
        Arrays.sort(nums);
        HashMap<Integer,List<List<Integer>>> hm= new HashMap<>();
        for(int i = 0; i < nums.length; i++) {
            if(i < nums.length-2 && nums[i] == nums[i+1] && nums[i] == nums[i+2]) continue;
            for(int j = i+1; j < nums.length; j++) {
                if(j < nums.length-1 && nums[j] == nums[j+1]) continue;
                int sum = nums[i] + nums[j];
                if(hm.containsKey(sum)) {
                    //如果包含sum
                    List<List<Integer>> cur = hm.get(sum);
                    boolean chongFu = true;
                    int m = 0;
                    for(m = 0; m < cur.size(); m++) {
                        List<Integer> c = cur.get(m);
                        if((nums[j] != nums[c.get(0)] && nums[j] != nums[c.get(1)])) {
                            //如果没有重复元素
                            chongFu = false;
                        }else {
                            //如果有重复的元素
                            chongFu = true;
                            break;
                        }
                    }
                    if(chongFu) {
                        cur.set(m,Arrays.asList(i,j));
                    }else {
                        cur.add(Arrays.asList(i, j));
                    }
                }else {
                    //如果不包含sum这个键，需要new对象加入
                    List<List<Integer>> r = new ArrayList<>();
                    r.add(Arrays.asList(i, j));
                    hm.put(sum, r);
                }
            }
        }
        for(int i = 0; i < nums.length; i++) {
            //去重
            if(i > 0 && nums[i] == nums[i-1]) continue;

            if(hm.containsKey(-nums[i])) {
                List<List<Integer>> cur = hm.get(-nums[i]);
                for(int j = 0; j < cur.size(); j++) {
                    List<Integer> c = cur.get(j);
                    if(i < c.get(0) && i < c.get(1)) {
                        //保证不会有重复的元素
                        res.add(Arrays.asList(nums[i],nums[c.get(0)],nums[c.get(1)]));
                    }
                }
            }
        }
        return res;
    }
}
```

时间复杂度：O（n^2^）

但是提交后发现超时，可能是因为对哈希表的查询也是较大的时间开销，所以不能用哈希表。

#### 解法二：双指针

使用双指针法来解决这道题。为了防止重复问题首先要将数组排列成有序的，这是许多对重复元素有要求的题目中的惯用伎俩。

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums.length < 3) return res;
        Arrays.sort(nums);
        for(int i = 0; i < nums.length-2; i++) {
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            int left = i + 1;
            int right = nums.length - 1;
            while(left < right) {
                int newSum = nums[i]+nums[left]+nums[right];
                if(newSum == 0) {
                    res.add(Arrays.asList(nums[i], nums[left++], nums[right--]));
                }else if(newSum > 0) {
                    right--;
                }else if(newSum < 0) {
                    left++;
                }
                while(left > i + 1 && left < nums.length-1 && nums[left] == nums[left - 1]) { 
                    //如果当前left指针指向元素与前一个元素相同那么要一直移动left直到不同
                    left++;
                }
                while(right < nums.length - 1 && right > left && nums[right] == nums[right + 1]) { //同理上面left指针的操作
                    right--;
                }
            }
        }
        return res;
    }
}
```

时间复杂度：O（n²）

空间复杂度：O（N），最坏情况，即 N 是指 n 个元素的排列组合个数，即 N=C^3^~n~，用来保存结果。

### 16. 最接近的三数之和

给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

 

**示例：**

```
输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
```

链接：https://leetcode-cn.com/problems/3sum-closest

#### 解法一：双指针

与15题一样的思路

```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        //sum保存结果，r保存绝对值差
        int sum = 0, r = Integer.MAX_VALUE;
        Arrays.sort(nums);
        for(int i = 0; i < nums.length - 2; i++) {
            int left = i+1, right = nums.length - 1;
            int curSum = 0;
            while(left < right) {
                curSum = nums[i]+ nums[left] + nums[right];
                int remain = curSum - target;
                if(remain == 0) {
                    return curSum;
                }else if(remain > 0){
                    if(remain < r) {
                        r = remain;
                        sum = curSum;
                    }
                    right--;
                }else {
                    if(-remain < r) {
                        r = -remain;
                        sum = curSum;
                    }
                    left++;
                }
            }
        }
        return sum;
    }
}
```

时间复杂度：O（n^2^）

空间复杂度：O（1）

#### 解法二：暴力解法

这题用暴力解法竟然也能通过。

```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        int sum = 0, r = Integer.MAX_VALUE;
        for(int i = 0; i < nums.length - 2; i++) {
            for(int j = i + 1; j < nums.length - 1; j++) {
                for(int m = j + 1; m < nums.length; m++) {
                    if(Math.abs(target - nums[i] - nums[j] - nums[m]) < r) {
                        r = Math.abs(target - nums[i] - nums[j] - nums[m]);
                        sum = nums[i] + nums[j] + nums[m];
                    }
                }
            }
        }
        return sum;
    }
}
```

时间复杂度：O（n^3^）

空间复杂度：O（1） 

### 17. 电话号码的字母组合

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/original_images/17_telephone_keypad.png" style="zoom:50%;" />

**示例 1：**

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**示例 2：**

```
输入：digits = ""
输出：[]
```

**示例 3：**

```
输入：digits = "2"
输出：["a","b","c"]
```

链接：https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number

#### 解法一：回溯法

这题是很典型的回溯法，所谓回溯法就是带有撤销状态的深度遍历法。

```java
class Solution {
    //用全局变量，是为了避免函数需要传递参数的麻烦
    List<String> result = new ArrayList<>();
    String[] str = {"!@#","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};//存储字母对应数字
    int len;
    StringBuilder path = new StringBuilder();
    public List<String> letterCombinations(String digits) {
        len = digits.length();
        if(len == 0) return result;
        backTracking(digits,0);
        return result;
    }
    private void backTracking(String digits, int depth) {
        if(depth == len){//当path存储的字符数达到要求就加入result中并弹栈
            //path数据类型是StringBuilder，必须要转成String
            result.add(path.toString());
            return;
        }
        //获取digits中的数字，减1的操作是为了方便str数组查找
        int startDigit = Character.getNumericValue(digits.charAt(depth)) -1;
        //str[startDigit].length()是该键对应的字符数
        for(int i = 0; i < str[startDigit].length(); i++) {
            //选取当前数字键的一个字母
            path.append(str[startDigit].charAt(i));
            //调用更深一层的递归
            backTracking(digits,depth+1);
            //撤销当前数字键选用的字母，进入下次循环
            path.setLength(path.length() - 1);
        }
    }
}
```

执行用时0 ms。

#### 解法二：队列迭代

既然可以用递归法解决这道题，那么相应的也可以尝试是否可以用迭代来解决。

```java
class Solution {
    public List<String> letterCombinations(String digits) {
        LinkedList<String> result = new LinkedList<>();
        int len = digits.length();
        if(len == 0) return result;
        result.add("");
        String[] str = {"!@#","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};//存储字母对应数字
        //将原digits字符串转换成char数组是为了方便快速取出
        char[] chars = digits.toCharArray();
        for(int i = 0; i < len; i++) {
            int num = Character.getNumericValue(chars[i]) - 1;
            while(result.peek().length() == i) {
                //取出队首元素
                String pre = result.remove(); 
                for(int j = 0; j < str[num].length(); j++) {
                    result.add(pre + str[num].charAt(j));
                }
            }
        }
        return result;
    }
}
```

实际提交代码的时间效率是不如上面的回溯法的，思考原因，有可能是因为力扣上的代码解析系统本身就会做递归更快，也有可能是这里用到的是字符串直接相加拼接而不是用StringBuilder做字符串拼接，时间开销会比较大。

于是用StringBuilder对代码作修改，如下。

```java
class Solution {
    public List<String> letterCombinations(String digits) {
        LinkedList<String> result = new LinkedList<>();
        int len = digits.length();
        if(len == 0) return result;
        result.add("");
        String[] str = {"!@#","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};//存储字母对应数字
        //将原digits字符串转换成char数组是为了方便快速取出
        char[] chars = digits.toCharArray();
        
        for(int i = 0; i < len; i++) {
            int num = Character.getNumericValue(chars[i]) - 1;
            while(result.peek().length() == i) {
                //取出队首元素
                StringBuilder path = new StringBuilder(result.remove());
                for(int j = 0; j < str[num].length(); j++) {
                    path.append(str[num].charAt(j));
                    result.add(path.toString());
                    path.deleteCharAt(path.length() - 1);
                }
            }
        }
        return result;
    }
}
```

明显速度快了很多，执行用时只用了1 ms。看来用StringBuilder和用String直接做字符串拼接速度还是差很多的。

### 18. 四数之和

给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

注意：答案中不可以包含重复的四元组。

**示例 1：**

```
输入：nums = [1,0,-1,0,-2,2], target = 0
输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
```

**示例 2：**

```
输入：nums = [], target = 0
输出：[]
```

链接：https://leetcode-cn.com/problems/4sum



这题和15题三数之和都是同样的思路同样的解法，用双指针，没啥好说的。

```java
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums.length < 4) return res;
        Arrays.sort(nums);
        
        for(int i = 0; i < nums.length; i++) {
            if(i > 0 && nums[i-1] == nums[i]) continue;
            for(int j = i+1; j < nums.length; j++) {                
                if(j > i+1 && nums[j] == nums[j-1]) continue;
                int sum = nums[i]+nums[j];
                int left = j+1, right = nums.length - 1;
                while(left < right) {
                    if(left > j + 1 && nums[left] == nums[left - 1]) {
                        left++;
                        continue;
                    }
                    if(right < nums.length - 1 && nums[right] == nums[right + 1]) {
                        right--;
                        continue;
                    }
                    int newSum = sum + nums[left] + nums[right];
                    if(newSum == target) {
                        res.add(Arrays.asList(nums[i], nums[j], nums[left++], nums[right--]));
                    }else if(newSum > target) {
                        right--;
                    }else{
                        left++;
                    }
                }
            }
        }
        return res;
    }
}
```

### 19. 删除链表的倒数第N个结点

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

**进阶：**你能尝试使用一趟扫描实现吗？

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

**示例 2：**

```
输入：head = [1], n = 1
输出：[]
```

**示例 3：**

```
输入：head = [1,2], n = 1
输出：[1]
```

链接：https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list



#### 解法一：顺序扫描两次

先扫描一遍得出总结点数，再进行第二次扫描找到sz-n的节点。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int sz = 0;
        ListNode p = head;
        while(p != null) {
            sz++;
            p = p.next;
        }
        if(sz == n) return head.next;
        int N = 1;
        p = head;
        while(N < sz-n) {
            N++;
            p = p.next;
        }
        p.next = p.next.next;
        return head;
    }
}
```

时间复杂度：O（sz）。

空间复杂度：O（1）。

#### 解法二：递归

上一种解法是两次正向的遍历，这种解法可以看成是一种触底回退操作。

```java
class Solution {
    int N = 0;
    public ListNode removeNthFromEnd(ListNode head, int n) {
        N = n;
        ListNode fakeHead = new ListNode(0, head);
        removeNthFromEnd(fakeHead);
        return fakeHead.next;
    }
    public ListNode removeNthFromEnd(ListNode head) {
        if(head == null) return head;
        head.next = removeNthFromEnd(head.next);
        if(--N == 0) {
            return head.next;
        }
        return head;
    }
}
```

时间复杂度：O（sz）。

空间复杂度：O（1）。

#### 解法三：双指针

上面两种解法执行时间都是0 ms，但是并不满足进阶要求中的“只遍历一次”。可以利用双指针达到要求。先将第一个指针移动n步，保证了第一个指针和第二个指针之间相差了n个，然后两个指针再一起同步移动，直到第一个指针移动到了队尾，此时第二个指针所在位置的下一个节点就是我们要删除的那个。

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode fakeHead = new ListNode(0,head);
        ListNode first = fakeHead, second = fakeHead;
        for(int i = 0; i < n; i++) {
            first = first.next;
        }
        while(first.next != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return fakeHead.next;
    }
}
```

时间复杂度：O（sz）。

空间复杂度：O（1）。

#### 解法四：空间换时间

看到一种相对来说比较骚的写法，就是遍历把每个节点都加入到一个列表中，然后直接取出在列表中位置为sz-n-1的节点。

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        List<ListNode> list = new ArrayList<ListNode>();
        int sz = 0;
        ListNode p = head;
        while(p != null) {
            list.add(p);
            sz++;
            p = p.next;
        }
        if(sz == n) return head.next;
        p = list.get(sz - n - 1);
        p.next = p.next.next;
        return head;
    }
}
```

时间复杂度：O（sz）。

空间复杂度：O（sz）。

不过实际提交的时候执行时间反而比前几种解法要差。

### 20. 有效的括号

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。

**示例 1：**

```
输入：s = "()"
输出：true
```

**示例 2：**

```
输入：s = "()[]{}"
输出：true
```

**示例 3：**

```
输入：s = "(]"
输出：false
```

**示例 4：**

```
输入：s = "([)]"
输出：false
```

**示例 5：**

```
输入：s = "{[]}"
输出：true
```

链接：https://leetcode-cn.com/problems/valid-parentheses

这道题利用栈就可以轻松解决

```java
class Solution {
    public boolean isValid(String s) {
        if(s.length()%2 != 0) return false;
        Stack<Character> st = new Stack<>();
        for(int i = 0; i < s.length(); i++) {
            char now = s.charAt(i);
            if(now == '(' || now == '{' || now == '[') {
                st.push(now);
            }else {
                if(st.isEmpty()) return false;
                if(now == ')') {
                    if(st.peek() != '(') return false;
                    st.pop();
                }else if(now == '}') {
                    if(st.peek() != '{') return false;
                    st.pop();
                }else if(now == ']') {
                    if(st.peek() != '[') return false;
                    st.pop();
                }
            }                        
        }
        return st.isEmpty();
    }
}
```

时间复杂度：O（n）。

空间复杂度：O（n）。

### 21. 合并两个有序链表

将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)

```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**示例 2：**

```
输入：l1 = [], l2 = []
输出：[]
```

**示例 3：**

```
输入：l1 = [], l2 = [0]
输出：[0]
```

链接：https://leetcode-cn.com/problems/merge-two-sorted-lists

#### 解法一：迭代

两条链表各有一指针，指向较小val值的指针节点接入当前要返回链表指针处。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        ListNode head;//要返回链表的头节点
        int num1 = l1.val, num2 = l2.val;
        //确定头节点是两个链表中的哪一个的头节点
        if(num1 <= num2) {
            head = new ListNode(num1);
            l1 = l1.next;
        }else {
            head = new ListNode(num2);
            l2 = l2.next;
        }
        
        ListNode p = head;
        while(l1 != null && l2 != null){
            num1 = l1.val;
            num2 = l2.val;
            //创建以较小值为val的ListNode连接到链表中
            if(num1 <= num2) {
                p.next = new ListNode(num1);
                l1 = l1.next;
            }else {
                p.next = new ListNode(num2);
                l2 = l2.next;
            }
            p = p.next;
        }
        //当一个链表到结尾时，另一个未必也同步到达结尾，需要将另一个的剩余部分连接到链表中
        if(l1 == null){
            p.next = l2;
        }else if(l2 == null) {
            p.next = l1;
        }
        return head;
    }
}
```

时间复杂度：O（n），n是两个链表的最短长度

空间复杂度：O（1）

上面的代码首先要确定头节点，代码显得多少有些啰嗦了，可以利用创建一个虚拟头节点来避免这样的麻烦。

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        //创建虚拟头节点
        ListNode fakeHead = new ListNode(0);
        int num1, num2;
        ListNode p = fakeHead;
        while(l1 != null && l2 != null){
            num1 = l1.val;
            num2 = l2.val;
            //创建以较小值为val的ListNode连接到链表中
            if(num1 <= num2) {
                p.next = new ListNode(num1);
                l1 = l1.next;
            }else {
                p.next = new ListNode(num2);
                l2 = l2.next;
            }
            p = p.next;
        }
        //当一个链表到结尾时，另一个未必也同步到达结尾，需要将另一个的剩余部分连接到链表中
        if(l1 == null){
            p.next = l2;
        }else if(l2 == null) {
            p.next = l1;
        }
        return fakeHead.next;
    }
}
```

时间复杂度：O（n），n是两个链表的最短长度

空间复杂度：O（1）

#### 解法二：递归

首先确定递归函数的终止条件是l1和l2中某一个为null。

如果都不是null，那么自然是要比较l1和l2的val值，以较小的值为val创建新节点，那么next是谁呢？这就要通过递归调用自身函数来获取了。

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        int num1 = l1.val, num2 = l2.val;
        if(num1 <= num2) {
            return new ListNode(num1, mergeTwoLists(l1.next, l2));
        }else {
            return new ListNode(num2, mergeTwoLists(l1,l2.next));
        }
    }
}
```

### 22. 括号生成

数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且有效的括号组合。

**示例 1：**

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
```

**示例 2：**

```
输入：n = 1
输出：["()"]
```

链接：https://leetcode-cn.com/problems/generate-parentheses

#### 解法一：回溯

一看到组合，就会想到用回溯算法了。

```java
class Solution {
    List<String> result = new ArrayList<>();//最后返回的结果
    int all;//需要生成的括号对数
    char[] ch;//存放括号
    public List<String> generateParenthesis(int n) {
        all=n;
        ch = new char[2*all];
        backTracking(0,0);
        return result;
    }
    private void backTracking(int i, int j){        
        if(i==all && j==all) {
            result.add(new String(ch));//可以直接用char数组来生成String
            return;
        }
        if(i<all) {
            ch[i+j] = '(';
            backTracking(i+1,j);
        }
        if(j<i) {
            ch[i+j] = ')';
            backTracking(i,j+1);
        }
    }
}
```

#### 解法二：分治

由于题目中说了n是大于等于1的，所以括号必然会有一对。题目可以看成需要知道这一对括号里有多少种组合，且这一对括号外又会有多少种组合，即（left）+ right，而left和right又都是用该函数来求解完成的，所以递归调用即可。

```java
class Solution {    
    public List<String> generateParenthesis(int n) {        
        List<String> result = new ArrayList<>();//最后返回的结果
        if(n == 0) {
            result.add("");
        }else {
            for(int i = 0; i < n; i++) {
                for(String left : generateParenthesis(i)) {
                    for(String right : generateParenthesis(n - 1 - i)) {
                        result.add("(" + left + ")" + right);
                    }
                }
            }
        }                
        return result;
    }
}
```

用String做字符串拼接过于耗时，改成StringBuilder之后执行时间从11 ms降至4 ms

```java
class Solution {    
    public List<String> generateParenthesis(int n) {        
        List<String> result = new ArrayList<>();//最后返回的结果
        if(n == 0) {
            result.add("");
        }else {
            for(int i = 0; i < n; i++) {
                for(String left : generateParenthesis(i)) {
                    for(String right : generateParenthesis(n - 1 - i)) {
                        StringBuilder sb = new StringBuilder("(");
                        sb.append(left);
                        sb.append(")");
                        sb.append(right);
                        result.add(sb.toString());
                    }
                }
            }
        }                
        return result;
    }
}
```

### 23. 合并K个升序链表

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

**示例 1：**

```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

**示例 2：**

```
输入：lists = []
输出：[]
```

**示例 3：**

```
输入：lists = [[]]
输出：[]
```

链接：https://leetcode-cn.com/problems/merge-k-sorted-lists

#### 解法一：纵向比较

就是一列一列的比较，创建以最小值为val的节点连接到要返回的链表上，并将值最小的那个节点指针移向它的下一个。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        int count = lists.length;
        if(count == 0) return null;
        if(count == 1) return lists[0];

        ListNode fakeHead = new ListNode(0);
        ListNode p = fakeHead;
        int min = Integer.MAX_VALUE, pos = -1;
        while(true) {
            for(int i = 0; i < count; i++) {
                if(lists[i] == null) continue;//遇到空节点就跳过这一轮循环
                if(lists[i].val < min) {
                    min = lists[i].val;
                    pos = i;//记录下最小值节点的位置
                }
            }
            if(pos == -1) break; //代表链表中节点全是null
            p.next = new ListNode(min);
            p = p.next;
            lists[pos] = lists[pos].next;
            //pos和min一定要复位
            pos = -1;
            min = Integer.MAX_VALUE;
        }
        return fakeHead.next;
    }
}
```

时间复杂度：O（kn），k是数组的长度，n是最长链表节点数。

空间复杂度： O（N），N 表示最终链表的长度

 看了其他人的解法，发现其实并不需要创建新的节点，改一下指针指向即可

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        int count = lists.length;
        if(count == 0) return null;
        if(count == 1) return lists[0];

        ListNode fakeHead = new ListNode(0);
        ListNode p = fakeHead;
        int min = Integer.MAX_VALUE, pos = -1;
        while(true) {
            for(int i = 0; i < count; i++) {
                if(lists[i] == null) continue;//遇到空节点就跳过这一轮循环
                if(lists[i].val < min) {
                    min = lists[i].val;
                    pos = i;//记录下最小值节点的位置
                }
            }
            if(pos == -1) break; //代表链表中节点全是null
            p.next = lists[pos];//此处作了修改，不再创建新节点
            p = p.next;
            lists[pos] = lists[pos].next;
            //pos和min一定要复位
            pos = -1;
            min = Integer.MAX_VALUE;
        }
        return fakeHead.next;
    }
}
```

时间复杂度和空间复杂度都是和上面一样的。

#### 解法二：暴力

遍历所有的链表，将所有的值加入ArrayList中，然后排序，再从小到大按顺序创建新节点连入链表。

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        int count = lists.length;
        if(count == 0) return null;
        if(count == 1) return lists[0];
        List<Integer> valList = new ArrayList<Integer>();
        for(ListNode l : lists) {
            while(l != null) {
                valList.add(l.val);
                l = l.next;
            }
        }
        Collections.sort(valList);
        ListNode fakeHead = new ListNode(0);
        ListNode p = fakeHead;
        for(Integer val : valList) {
            p.next = new ListNode(val);
            p = p.next;
        }
        return fakeHead.next;
    }
}
```

时间复杂度： O（NlogN）， N 是所有值的个数，排序如果是用快速排序就是 O(NlogN)

空间复杂度：O（N）。

#### 解法三：分治

可以将链表两两合并，利用之前21题合并两个链表的代码。先以步长为1两两合并列表，再以步长为2，再以步长为4……

<img src="https://windliang.oss-cn-beijing.aliyuncs.com/23_4.jpg" style="zoom:80%;" />

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        int count = lists.length;
        if(count == 0) return null;
        if(count == 1) return lists[0];

        int step = 1;
        while(count > step) {
            for(int i = 0; i + step < lists.length; i+=(step*2)) {
                lists[i] = mergeTwoLists(lists[i], lists[i+step]);
            }
            step *= 2;
        }
        return lists[0];
    }
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        int num1 = l1.val, num2 = l2.val;
        if(num1 <= num2) {
            return new ListNode(num1, mergeTwoLists(l1.next, l2));
        }else {
            return new ListNode(num2, mergeTwoLists(l1,l2.next));
        }
    }
}
```

时间复杂度：O（n logk）合并 log（k）次，每次合并需要的时间复杂度是O（n）

空间复杂度：O（N）

### 26. 删除有序数组中的重复项

给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

**示例 1：**

```
输入：nums = [1,1,2]
输出：2, nums = [1,2]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
```

**示例 2：**

```
输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。
```

链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array

可以理解为设置了快慢两个指针，快指针一直向前移动，慢指针只有在快指针遇到了不同值时才向前移动，并将快指针处的值作修改。

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        if(nums.length == 0) return 0;
        if(nums.length == 1) return 1;
        int prev = nums[0], count = 1;
        for(int i = 1; i < nums.length; i++) {
            if(nums[i] != prev){
                nums[count] = nums[i];
                count++;
                prev = nums[i];
            }
        }
        return count;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 27. 移除元素

给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

 **示例 1：**

```
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。
```

**示例 2：**
```
输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
解释：函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。注意这五个元素可为任意顺序。你不需要考虑数组中超出新长度后面的元素。
```
链接：https://leetcode-cn.com/problems/remove-element

#### 解法一

这一题和上面是非常像了，用上一题的思路可以直接掰扯出代码。

```java
class Solution {
    public int removeElement(int[] nums, int val) {
        int low = 0;
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] != val) {
                nums[low++] = nums[i];
            }            
        }
        return low;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法二

因为题目说元素顺序可以不一样，所以可以通过不停地交换元素来实现。

一个指针 i 进行数组遍历，另外一个指针 j 指向有效数组的最后一个位置。

只有当 i 所指向的值和 j 不一致（不重复），才将 i 的值添加到 j 的下一位置。

[参考题解](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/solution/shua-chuan-lc-jian-ji-shuang-zhi-zhen-ji-2eg8/)

```java
class Solution {
    public int removeElement(int[] nums, int val) {
        int j = nums.length - 1;
        for(int i = 0; i <= j; i++) {
            if(nums[i] == val) {
                nums[i--] = nums[j--];
            }            
        }
        return j + 1;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 31. 下一个排列

实现获取 下一个排列 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须 原地 修改，只允许使用额外常数空间。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：nums = [3,2,1]
输出：[1,2,3]
```

**示例 3：**

```
输入：nums = [1,1,5]
输出：[1,5,1]
```

**示例 4：**

```
输入：nums = [1]
输出：[1]
```

链接：https://leetcode-cn.com/problems/next-permutation

这题当时写的时候在纸上乱写乱画找了很久的规律，以一个例子来解释思路吧。

以142635为例，首先要找到以数组尾部为结尾的降序子区间，也就是5，因为3是比5小的。然后降序子区间的前一个值是3，要往后找比3大的数并与他交换，就得到了结果142653。

以124653为例，找到降序子区间是653，将它倒序是356变成升序区间，然后该区间内比4大的值是5，将它们交换，就得到了125346。

```java
class Solution {
    public void nextPermutation(int[] nums) {
        int reverse = findreverse(nums);
        int left = reverse, right = nums.length - 1;
        while(left < right) {
            //逆转降序区间，这里其实也可以直接用Arrays.sort来解决
            swap(nums, left, right);
            left++;
            right--;
        }
        int i = reverse, prev = reverse - 1;
        if(prev >= 0) {
            //原降序区间前一个值与逆转顺序后的原降序区间现升序区间中第一个比自己大的数交换位置
            while(nums[i] <= nums[prev]) i++;
            swap(nums,i,prev);
        }
    }
    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
    private int findreverse(int[] nums) { 
        //找到数组尾部的降序子区间
        for(int i = nums.length - 1; i >= 1; i--) {
            if(nums[i] > nums[i - 1]) return i;
        }
        return 0;
    }
}
```

时间复杂度：O（n）。

空间复杂度：O（1）。

### 32. 最长有效括号

给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

**示例 1：**

```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```

**示例 2：**

```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

**示例 3：**

```
输入：s = ""
输出：0
```

链接：https://leetcode-cn.com/problems/longest-valid-parentheses

#### 解法一：暴力破解

很暴力，判断每一个子串，是否符合配对。

但是这种算法必然会超时。

#### 解法二：优化暴力破解

其实并不需要对每个子串都检查，例如一段2-7区间是有效括号配对的，如果按照解法一，我们必然会判断2-3、2-5、2-7，但是其实只需要在判断2-3发现的确有效后再往后接着判断就行。

```java
class Solution {
    public int longestValidParentheses(String s) {
        int len = s.length();
        if(len <= 1) return 0;
        char[] array = s.toCharArray();
        int max = 0;
        for(int i = 0; i < len; i++) {
            int count = 0;
            for(int j = i; j < len; j++) {
                if(array[j] == '(') {
                    count++;
                }else {
                    count--;
                }
                if(count < 0) {
                    break;
                }else if(count == 0){
                    max = Math.max(max,j-i+1);
                }
            }
        }
        return max;
    }
}
```

时间复杂度：O（n^2^）

空间复杂度：O（n）

#### 解法三：动态规划

首先明确动态规划的数组代表什么，dp[i]表示以 i 结尾有多少对有效括号。

假如 i 位置是左括号，那么dp[i] = 0

如果 i 位置是右括号，要看前一位i-1位置是什么。如果是左括号，那么i和i-1是一对有效的括号对，还要看i-2位置有多少对，得到dp[i] = dp[i-2] + 2。如果是右括号，就要看i-1-dp[i-1]位置是否为左括号。

以下图为例

<img src="https://windliang.oss-cn-beijing.aliyuncs.com/32_1.jpg" style="zoom:90%;" />

index = 3的位置，检查前一位发现是'('，所以再往前看一位有多少个有效括号，再加上2得到结果4。

index = 7的位置，前一位是')'，dp[6] = 2，所以再从6往前看两位，发现是个左括号，可以与7的右括号配对上，所以已经确定了从i-1-dp[i-1]到i有多少个配对了，但是前面可能还有可以配对的，这时候就要再加上dp[i-2-dp[i-1]] = dp[3] = 4，得到结果8。

```java
class Solution {
    public int longestValidParentheses(String s) {
        int len = s.length();
        if(len <= 1) return 0;
        char[] array = s.toCharArray();
        int[] dp = new int[len];
        int max = 0;
        for(int i = 0; i < len; i++) {
            if(array[i] == ')') {
                if(i == 0) continue;
                if(array[i-1] == '(') {
                    dp[i] = (i > 1 ? dp[i-2] : 0) + 2;
                }else if(i-1-dp[i-1] >= 0 && array[i-1-dp[i-1]] == '('){
                    dp[i] = ( i-1-dp[i-1] > 0 ? dp[i-2-dp[i-1]] : 0) + dp[i-1] + 2;
                }
                max = Math.max(max,dp[i]);
            }
        }
        return max;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

#### 解法四：栈

将左括号的位置都压入栈，一旦遇到右括号就弹栈，得到长度。如果弹栈后发现栈空了，就要把当前位置压入栈。

即保证每次弹栈后，栈顶的元素存入的是”以当前遍历所在位置为结尾的有效配对子串“可以开始的位置的-1位置（方便计算不用再+1了）。

```java
class Solution {
    public int longestValidParentheses(String s) {
        int len = s.length();
        if(len <= 1) return 0;
        char[] array = s.toCharArray();
        int max = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for(int i = 0; i < len; i++) {
            if(array[i] == '(') {
                stack.push(i);
            }else {
                stack.pop();
                if(stack.isEmpty()) {
                    stack.push(i);
                }else {
                    max = Math.max(max,i-stack.peek());
                }
            }
        }
        return max;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

#### 解法五：正序逆序双重扫描法

分别用left和right来记录左括号和右括号的数量。如果相等就更新max的值。

先正向扫描，一旦出现右括号数大于左括号数的情况，就将左右括号数都置为0。正向扫描一遍后再反向扫描，此时，一旦出现左括号数大于右括号数的情况，就将左右括号数都置为0。

之所以要正序反序都来一次，以“（（）”为例，正向扫描left永远大于right，这时候就需要反向来达到right = left的条件。

```java
class Solution {
    public int longestValidParentheses(String s) {
        int len = s.length();
        if(len <= 1) return 0;
        char[] array = s.toCharArray();
        int max = 0, left = 0, right = 0;
        for(int i = 0; i < len; i++) {
            if(array[i] == '(') {
                left++;
            }else {
                right++;
            }
            if(left == right) {
                max = Math.max(max, 2*left);
            }else if(right > left) {
                left = 0;
                right = 0;
            }
        }
        right = 0;
        left = 0;
        for(int i = len-1; i >= 0; i--) {
            if(array[i] == '(') {
                left++;
            }else {
                right++;
            }
            if(left == right) {
                max = Math.max(max, 2*left);
            }else if(right < left) {
                left = 0;
                right = 0;
            }
        }
        return max;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

### 33. 搜索旋转排序数组

整数数组 nums 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 **旋转**，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 **从 0 开始计数**）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

 **示例 1：**

```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```

**示例 2：**

```
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
```

**示例 3：**

```
输入：nums = [1], target = 0
输出：-1
```

链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array

#### 解法一：暴力破解法

这个方法就是拿来骚的，没啥用。

```java
class Solution {
    public int search(int[] nums, int target) {
        int i = 0;
        for (int num : nums) {
            if (num == target) {
                return i;
            }
            i++;
        }
        return -1;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法二：二分法

题目要求了时间复杂度为O(log n)，那么必然要用二分法来写了。

可以知道该数组就是被分隔成了两个部分，左边和右边都是升序的，且左边的值一定都是比右边的值大的。

用二分法设置left、right、mid三个指针，如果nums[left] < nums[right]，则代表当前搜索的区间都是升序的，做正常地二分搜索即可。

如果nums[left] > nums[right]就知道该区间内一定存在着一个断点。同时，如果nums[mid] > nums[left]，那么代表断点一定是mid右边，此时如果nums[mid] < target，就要去mid右边找，大于target不能确定到底在哪一边，还需要拿target和left与right的值作比较：如果target < nums[left]，则target在断点到right之间，去右边找，如果target >= nums[left]，去左边找。

如果nums[mid] < nums[left]，断点一定是mid左边。此时如果nums[mid] > target，target如果有的话那肯定是在断点和mid之间，去左边找。如果nums[mid] < target，不能确定到底在哪一边，还需要拿target和left与right的值作比较：如果target < nums[left]，去右边找，如果target >= nums[left]，去左边找。

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left <= right) {
            int mid = left + (right - left)/2;
            int nm = nums[mid], nl = nums[left], nr = nums[right];
            if(nums[mid] == target) return mid;
            if(nm < nl) {//mid在断点的右段
                if(target > nm && target <= nr) {
                    left = mid + 1;
                }else {
                    right = mid - 1;
                }
            } else {//mid在断点的左段
                if(target >= nl && target < nm) {
                    right = mid - 1;
                }else {
                    left = mid + 1;
                }
            }
        }
        return -1;
    }
}
```

时间复杂度：O（log n）

空间复杂度：O（1）

### 34. 在排序数组中查找元素的第一个和最后一个位置

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

进阶：

你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？

**示例 1：**

```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

**示例 2：**

```
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```

**示例 3：**

```
输入：nums = [], target = 0
输出：[-1,-1]
```

链接：https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array

#### 解法一：暴力破解

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] pos = {-1, -1};
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] == target && ((i >= 1 && nums[i-1] != target) || i == 0)) pos[0] = i;
            if(nums[i] == target && ((i < nums.length - 1 && nums[i+1] != target) || i == nums.length - 1)) {
                pos[1] = i;
                break;
            }
        }
        return pos;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法二：二分法

要求时间复杂度O（log n)，那么必定是用二分法了。

最开始的想法是用二分法找到一个target所在的位置，然后分别往左往右逐位找起始和末位点，但是这样的方法，如果遇到一整个数组都是target的情况，就会变成O（n）的时间复杂度。所以应该每次搜索都用二分搜索。

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int startP = 0;
        int endP = nums.length - 1;
        int[] pos = {-1, -1};
        if(endP == -1) return pos;
        int firstP = binarySearch(0, endP, target, nums);
        //如果搜索无果，代表根本不含有target，返回[-1,-1]
        if(nums[firstP] != target) return pos;
        
        int prevLp = firstP;//上次搜索到的区间起始点
        int prevRp = firstP;//上次搜索到的区间末尾点
        int nowP;//保存当前二分搜索返回的坐标
        
        //找区间起始点
        while (true) {
            nowP = binarySearch(0, prevLp - 1, target, nums);//注意要-1
            if(nums[nowP] == target) {
                if(nowP != prevLp) {
                    prevLp = nowP;
                } else{
                    break;
                }
            }else {
                break;
            }
        }
        //找区间结束点
        while (true) {
            nowP = binarySearch(prevRp + 1, endP, target, nums);
            if(nums[nowP] == target) {
                if(nowP != prevRp) {
                    prevRp = nowP;
                } else{
                    break;
                }
            }else {
                break;
            }
        }
        pos[0] = prevLp;
        pos[1] = prevRp;
        return pos;
    }

    private int binarySearch(int left, int right, int searchFor, int[]nums) {       
        while(left <= right) {
            int mid = left + (right - left)/2;
            int now = nums[mid];
            if(now == searchFor) {
                return mid;
            }else if(now < searchFor) {
                left = mid + 1;
            }else {
                right = mid - 1;
            }
        }
        return right >= 0 ? right: 0;
    }
}
```

时间复杂度：O（log n）

空间复杂度：O（1）


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

#### 解法：双指针

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

#### 解法：双指针

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

### 28. 实现 strStr()

实现 [strStr()](https://baike.baidu.com/item/strstr/811469) 函数。给你两个字符串 `haystack` 和 `needle` ，请你在 `haystack` 字符串中找出 `needle` 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回 `-1` 。

**说明：**

当 `needle` 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。对于本题而言，当 `needle` 是空字符串时我们应当返回 0 。这与 C 语言的 [strstr()](https://baike.baidu.com/item/strstr/811469) 以及 Java 的 [indexOf()](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html#indexOf(java.lang.String)) 定义相符。

**示例 1：**

```
输入：haystack = "hello", needle = "ll"
输出：2
```

**示例 2：**

```
输入：haystack = "aaaaa", needle = "bba"
输出：-1
```

**示例 3：**

```
输入：haystack = "", needle = ""
输出：0
```

#### 解法一：暴力法

两个字符串从头开始匹配，一旦遇到了不同的值，就回溯再从头开始找。

```java
class Solution {
    public int strStr(String haystack, String needle) {
        if(needle.length() == 0) return 0;
        if(needle.length() > haystack.length()) return -1;
        char[] hay = haystack.toCharArray(), need = needle.toCharArray();
        for(int i = 0; i <= haystack.length()-needle.length(); i++) {
            if(hay[i] != need[j]) continue;
            int tmp = i;
            int j = 0;
            while(j < needle.length()) {
                if(hay[i] != need[j]) break;
                i++;
                j++;
            }
            if(j == needle.length()) return tmp;
            i = tmp;
        }
        return -1;
    }
}
```

时间复杂度：O（m*n）

#### 解法二：KMP

[具体看](https://www.cnblogs.com/dusf/p/kmp.html)

如果是人为来寻找的话，肯定不会再把i移动回第1位，**因为主串匹配失败的位置前面除了第一个A之外再也没有A了**，我们为什么能知道主串前面只有一个A？**因为我们已经知道前面三个字符都是匹配的！（这很重要）**。移动过去肯定也是不匹配的！有一个想法，**i**可以不动，我们只需要移动**j**即可，如下图：

 ![img](https://images0.cnblogs.com/blog/416010/201308/17083828-cdb207f5460f4645982171e58571a741.png)

上面的这种情况还是比较理想的情况。但假如是在主串“SSSSSSSSSSSSSA”中查找“SSSSB”，比较到最后一个才知道不匹配，然后i回溯，这个的效率是显然是最低的。

所以需要创建一个`next[i]`数组，当`haystack[j] != needle[i]`的时候，`i`指针要移到`next[i-1]`的位置。

![](https://images0.cnblogs.com/blog/416010/201308/17083929-a9ccfb08833e4cf1a42c30f05608f8f5.png)

以上面的图为例，从头开始比较到第四个字符的时候，发现C和D不同，所以指向`needle`的`j`指针要移动到一个满足 “ 前缀字符子串 与 当前`haystack`的`i`指针指向位置 前面一小段子串 相同” 的位置。发现C前面的字符子串"A"和`needle`的前缀字符串“A”是相同的，所以将`j`移动到“B"的位置。

知道了`next[i]`的作用，具体该如何求是个问题。

- 特殊情况：

  初始化设定`next[0]=-1`，因为`next[i]`是在前面`i`个字符都匹配的情况下唯独索引位置在`i`的位置不匹配时需要移动指针所达的位置，`next[0]`代表前面有`0`个字符相匹配，自然初始化为一个非法下标。

  `next[1]=0`，因为前面只有一个字符相匹配，而当前这个字符不匹配，只能将其移动到初始位置。

- `t[j] == t[k]` 的情况：

  举个栗子![](https://img-blog.csdnimg.cn/20190322134050593.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RhcmtfY3k=,size_16,color_FFFFFF,t_70)观察上图可知，当 t[j] == t[k] 时，必然有`t[0…k-1]" == " t[j-k…j-1]`，此时的 k 即是相同子串的长度。因为` t[j] == t[k]`，则有"t[0]…t[k]" == " t[j-k]…t[j]"，这样也就得出了next[j+1]=k+1。

- 当`t[j] != t[k]` 的情况：

  看下图：

  ![](https://img-blog.csdnimg.cn/20190322194629884.png)

  能够保证的是，`j`前面的一段`[j-k...j-1]`与`[0...k-1]`的子串是一样的，对应上图就是：此时`k`前面的ABA与`j`前面的ABA是重合的，求`next[j]`就相当于求 `next[k]`，因此`next[j] = next[k]`。

```java
class Solution {
    public int strStr(String haystack, String needle) {
        if(needle.length() == 0) return 0;
        if(needle.length() > haystack.length()) return -1;
        char[] h = haystack.toCharArray(), n = needle.toCharArray();        
        int[] next = new int[n.length];
        getNext(n, next);
        int i = 0, j = 0;
        while(i < h.length && j < n.length) {
            if(j == -1 || h[i] == n[j]) {
                j++;
                i++;
            }else {
                j = next[j];
            }
        }
        if(j == n.length) return i-n.length;
        return -1;
    }
    private void getNext(char[] pattern, int[] next) {
        next[0] = -1;
        int j = 0, k = -1;
        while(j < pattern.length-1) {
            if(k == -1 || pattern[j] == pattern[k]) {
                j++;
                k++;
                if(pattern[k] == pattern[j]) {//当两个字符相同时特殊处理
                    next[j] = next[k];
                }else {
                    next[j] = k;
                }                
            }else {
                k = next[k];
            }
        }
    }
}
```



#### 解法三：内置函数

```java
class Solution {
    public int strStr(String haystack, String needle) {
        return haystack.indexOf(needle);
    }
}
```

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

#### 解法：找规律

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

### 35. 搜索插入位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

**示例 1:**

```
输入: [1,3,5,6], 5
输出: 2
```

**示例 2:**

```
输入: [1,3,5,6], 2
输出: 1
```

**示例 3:**

```
输入: [1,3,5,6], 7
输出: 4
```

**示例 4:**

```
输入: [1,3,5,6], 0
输出: 0
```

链接：https://leetcode-cn.com/problems/search-insert-position

#### 解法：二分查找

首先可以肯定的是这题一定会用到二分法来搜寻，最开始一个最清晰易理解的思路就是，二分查找过程中如果遇到了nums[mid]等于target值就直接返回mid。

如果nums[mid] < target，往后看一位，如果后一位刚好大于target或已经达到了数组尾，就代表mid+1位置是应该插入数字的地方。如果后一位不满足上述要求，那么肯定还要往mid的右边继续二分查找。

如果nums[mid] > target，往前看一位，如果前一位刚好小于target或已经达到了数组头，就代表mid所在位置是应该插入数字的地方。如果不满足上述要求，还要往左边继续二分查找。

```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        if(right == 0) return target > nums[0] ? 1 : 0;
        if(right == -1) return 0;
        
        while(left <= right) {
            int mid = left + (right - left)/2;
            if(nums[mid] == target) return mid;
            if(nums[mid] < target) {
                //mid后一个数大于target，或者mid已经是最后一个数
                //直接返回mid+1
                if((mid + 1 < nums.length && nums[mid+1] > target) || mid == nums.length - 1) return mid+1;
                //不满足上述条件还要继续往右
                left = mid + 1;
            }else{
                //mid前一个数小于target，或者mid已经是第一个数
                //直接返回mid
                if((mid > 0 && nums[mid-1] < target) || mid == 0) return mid;
                //不满足上述条件还要继续往左
                right = mid - 1;
            }
        }
        return -1;
    }
}
```

时间复杂度：O（log n）

空间复杂度：O（1）

但其实中间的那些if判断是不用做的，可以直接无脑做二分查找，如果最后没有在left到right区间查到target，跳出while循环时，left就是我们应该插入target的返回值。

其实也是好理解的，因为每次缩短搜索区间的时候，都保证了，left左边的数一定都小于target，right右边的数一定都大于target，最后跳出while循环时，right>left，这代表，right右边包括left在内的所有值都大于target，left左边包括right在内的所有值都小于target，target要添加在比它小的数字后面（即一定在”left左边所有数字“的右边），比它大的数字前面（即”从left开始往右的所有数字“的左边），所以它就该插入在left。

```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        if(right == 0) return target > nums[0] ? 1 : 0;
        if(right == -1) return 0;
        
        while(left <= right) {
            int mid = left + (right - left)/2;
            if(nums[mid] == target) return mid;
            if(nums[mid] < target) {
                left = mid + 1;
            }else{
                right = mid - 1;
            }
        }
        return left;
    }
}
```

时间复杂度：O（log n）

空间复杂度：O（1）

### 36. 有效的数独

请你判断一个 9x9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
数独部分空格内已填入了数字，空白格用 '.' 表示。

注意：

一个有效的数独（部分已被填充）不一定是可解的。
只需要根据以上规则，验证已经填入的数字是否有效即可。

**示例 1：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/04/12/250px-sudoku-by-l2g-20050714svg.png)

```
输入：board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
输出：true
```

**示例 2：**

```
输入：board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
输出：false
解释：除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。 但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。
```

链接：https://leetcode-cn.com/problems/valid-sudoku

#### 解法一：

根据题目有三个要求，分别是行、列、3x3宫。那么最简单的想法就是先对每一行扫描，看是否有重复数字，再对每一列扫描，看是否有重复数字，再对每个宫查找。

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {    
        //看每一行    
        for(int i = 0; i < 9; i++) {
            //每一行开始扫描时都重新创建shown数组用来记录每一行内该数字是否出现过
            //例如 shown[8] = true，代表在这一行内8已经出现过了
            boolean[] shown = new boolean[9];
            for(int j = 0; j < 9; j++) {
                char c = board[i][j];
                if(c == '.') continue;
                if(shown[c - '1']) return false;
                shown[c - '1'] = true;
            }
        }
        //看每一列
        for(int i = 0; i < 9; i++) {
            boolean[] shown = new boolean[9];
            for(int j = 0; j < 9; j++) {
                char c = board[j][i];
                if(c == '.') continue;
                if(shown[c - '1']) return false;
                shown[c - '1'] = true;
            }
        }
        //看每个宫
        for(int i = 0; i < 9; i++) {
            boolean[] shown = new boolean[9];
            //i表示是第几个宫
            for(int j = 0; j < 9; j++) {
                //j表示是宫内第几个格子
                char c = board[(i/3)*3 + j/3][(i%3)*3 + j%3];
                if(c == '.') continue;
                if(shown[c - '1']) return false;
                shown[c - '1'] = true;
            }
        }
        return true;
    }
}
```

时间复杂度：整个盘遍历了三次，3n，所以复杂度是O（n）

空间复杂度：O（1）

#### 解法二：

上面的解法，必然是做了重复搜索的。可以用一种空间换时间的方法。分别用三个数组，记录数字在行、列、宫是否出现过。这样整个盘只需要遍历一次。

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        //表示某个数字在某一行出现过
        //例如row[2][8] = true表示在第2行数字9出现过
        boolean[][] row = new boolean[9][9];
        //表示某个数字在某一列出现过
        //例如col[2][8] = true表示在第2列数字9出现过
        boolean[][] col = new boolean[9][9];
        //表示某个数字在某个3*3宫内出现过
        //把整个板看成是3*3的大宫，每个大宫里有3*3小宫
        //例如singleB[0][2][7] = true表示在大宫第0行第2列内数字8出现过
        boolean[][][] sigleB = new boolean[3][3][9];
        for(int i = 0; i < 9; i++) {
            for(int j = 0; j < 9; j++) {
                char c = board[i][j];
                if(c == '.') continue;
                if(row[i][c - '1'] || col[j][c - '1'] || sigleB[i/3][j/3][c - '1']) return false;
                row[i][c - '1'] = true;
                col[j][c - '1'] = true;
                sigleB[i/3][j/3][c - '1'] = true;
            }
        }
        return true;
    }
}
```

时间复杂度：O（n），复杂度其实和上面的是一样的。

空间复杂度：O（1）

实际提交的时候，解法一和解法二的执行时间一样。

### 37. 解数独

编写一个程序，通过填充空格来解决数独问题。

数独的解法需 遵循如下规则：

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
数独部分空格内已填入了数字，空白格用 '.' 表示。

**示例：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/04/12/250px-sudoku-by-l2g-20050714svg.png)

```
输入：board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
输出：[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
解释：输入的数独如上图所示，唯一有效的解决方案如下所示：
```

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/04/12/250px-sudoku-by-l2g-20050714_solutionsvg.png)

链接：https://leetcode-cn.com/problems/sudoku-solver

#### 解法：回溯法

首先这一题可以利用上一题用三个数组分别存储行、列、宫是否含有某个数字的思想。

然后，解数独是回溯法中一种非常典型的题型，可以确定这题用回溯法。

确定返回条件，如果遇到了某个格子填入任何值都不满足题目中的三个要求就需要返回。

那么是要将整个盘子的遍历都放在递归函数里吗？即，递归函数中，从传入的坐标开始往后一个个搜索，如果不是'.'，就做正常判断，不满足要求就返回，满足要求就将相应的标记数组修改状态，再继续往下。如果是'.'，就对1-9每个数进行填入尝试，如果暂且满足这个格子的操作，就递归下一个格子。

```java
class Solution {
    private boolean[][] row = new boolean[9][9];//若row[0][8]=true表明在0行数字9已经出现过
    private boolean[][] col = new boolean[9][9];//若col[0][8]=true表明在0列数字9已经出现过
    private boolean[][][] block = new boolean[3][3][9];//若block[0][1][8]=true表明在0行1列的板块中出现过数字8
    
    public void solveSudoku(char[][] board) {
        dfs(board, 0, 0);
    }
    public void dfs(char[][] board, int i, int j) {
        char c = board[i][j];
        if(c != '.') {
            if(row[i][c-'1'] || col[j][c-'1'] || block[i/3][j/3][c-'1']) return;
            if(j <= 7) {
                dfs(board,i,j+1);
            }else if(i <= 7) {
                 dfs(board,i+1,0);
            }
        }else {
            boolean end = false;
            for(int n = 0; n <= 8; n++) {
                if(row[i][n] || col[j][n] || block[i/3][j/3][n]) continue;
                row[i][n] = col[j][n] = block[i/3][j/3][n] = true;
                board[i][j] = (char)(n + '1');
                if(j <= 7) {
                    dfs(board,i,j+1);
                    row[i][n] = col[j][n] = block[i/3][j/3][n] = false;
                }else if(i <= 7) {
                    dfs(board,i+1,0);
                    row[i][n] = col[j][n] = block[i/3][j/3][n] = false;
                }
                if(i == 8 && i == 8) {
                    end = true;
                    break;
                }
            }
            if(!end) {
                board[i][j] = '.';
                return;
            }
        }
    }
}
```

解法一超出了时间限制。

想想就知道了，比如说遇到一个比较靠前的'.'，对它这个格子的约束还非常少，可以添加的值比较多，但是向后遍历的时候，就会遇到各种约束出现了，这时可能已经递归到了相当深的地方，消耗很多时间。

所以可以先在主函数中对整个盘子进行一次遍历，对三个布尔数组做好标记，记录下需要填入数字的那些格子坐标。然后再去调用递归，只对那些需要填写数字的格子进行递归，这样就不会陷入一个很深的地方。

```java
class Solution {
    private boolean[][] row = new boolean[9][9];//若row[0][8]=true表明在0行数字9已经出现过
    private boolean[][] col = new boolean[9][9];//若col[0][8]=true表明在0列数字9已经出现过
    private boolean[][][] block = new boolean[3][3][9];//若block[0][1][8]=true表明在0行1列的板块中出现过数字8
    private boolean valid = false;    
    private List<int[]> spaces = new ArrayList<int[]>();//记录需要填入数字的坐标
    public void solveSudoku(char[][] board) {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (board[i][j] == '.') {
                    spaces.add(new int[]{i, j});
                } else {
                    int digit = board[i][j] - '0' - 1;
                    row[i][digit] = col[j][digit] = block[i / 3][j / 3][digit] = true;
                }
            }
        }
        dfs(board, 0);
    }
    public void dfs(char[][] board, int pos) {
        if (pos == spaces.size()) {
            valid = true;
            return;
        }
        int[] space = spaces.get(pos);
        int i = space[0], j = space[1];
        for (int digit = 0; digit < 9 && !valid; ++digit) {
            if (!row[i][digit] && !col[j][digit] && !block[i / 3][j / 3][digit]) {
                row[i][digit] = col[j][digit] = block[i / 3][j / 3][digit] = true;
                board[i][j] = (char) (digit + '0' + 1);
                dfs(board, pos + 1);
                row[i][digit] = col[j][digit] = block[i / 3][j / 3][digit] = false;
            }
        }
    }
}
```

### 38. 外观数列

给定一个正整数 n ，输出外观数列的第 n 项。

「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。

你可以将其视作是由递归公式定义的数字字符串序列：

countAndSay(1) = "1"
countAndSay(n) 是对 countAndSay(n-1) 的描述，然后转换成另一个数字字符串。
前五项如下：

1.     1
2.     11
3.     21
4.     1211
5.     111221
第一项是数字 1 
描述前一项，这个数是 1 即 “ 一 个 1 ”，记作 "11"
描述前一项，这个数是 11 即 “ 二 个 1 ” ，记作 "21"
描述前一项，这个数是 21 即 “ 一 个 2 + 一 个 1 ” ，记作 "1211"
描述前一项，这个数是 1211 即 “ 一 个 1 + 一 个 2 + 二 个 1 ” ，记作 "111221"
要 描述 一个数字字符串，首先要将字符串分割为 最小 数量的组，每个组都由连续的最多 相同字符 组成。然后对于每个组，先描述字符的数量，然后描述字符，形成一个描述组。要将描述转换为数字字符串，先将每组中的字符数量用数字替换，再将所有描述组连接起来。

例如，数字字符串 "3322251" 的描述如下图：

![](https://assets.leetcode.com/uploads/2020/10/23/countandsay.jpg)

**示例 1：**

```
输入：n = 1
输出："1"
解释：这是一个基本样例。
```

**示例 2：**

```
输入：n = 4
输出："1211"
解释：
countAndSay(1) = "1"
countAndSay(2) = 读 "1" = 一 个 1 = "11"
countAndSay(3) = 读 "11" = 二 个 1 = "21"
countAndSay(4) = 读 "21" = 一 个 2 + 一 个 1 = "12" + "11" = "1211"
```

链接：https://leetcode-cn.com/problems/count-and-say

#### 解法一：迭代

直接按照题意所说的，从1开始，一个个描述，直到描述到了n。

```java
class Solution {
    public String countAndSay(int n) {
        StringBuilder pre = new StringBuilder("1");
        StringBuilder cur = new StringBuilder();
        while(n-- > 1) {
            char c = pre.charAt(0);
            int num = 1;
            for(int i = 1; i <= pre.length(); i++) {
                if(i == pre.length()) {
                    //注意到超出数组边界时要把最后的结果加入
                    cur.append((char)('0'+num));
                    cur.append(c);
                }else {
                    if(pre.charAt(i) == c) {
                        num++;
                    }else {
                        cur.append((char)('0'+num));
                        cur.append(c);
                        c = pre.charAt(i);
                        num = 1;
                    }
                }                
            }
            pre = cur;
            cur = new StringBuilder();
        }
        return pre.toString();
    }
}
```

#### 解法二：递归

通过调用自身函数来获得前一个数字的外观数列。

```java
class Solution {
    public String countAndSay(int n) {
        if(n == 1) return "1";
        StringBuilder cur = new StringBuilder();
        //通过递归调用获取上一个数字的外观数列
        String pre = countAndSay(n-1);
        char[] chars = pre.toCharArray();
        int num = 1;
        char c = pre.charAt(0);
        for(int i = 1; i <= chars.length; i++) {
            if(i == chars.length) {
                cur.append((char)('0'+num));
                cur.append(c);
            }else {
                if(chars[i] == c) {
                    num++;
                }else {
                    cur.append((char)('0'+num));
                    cur.append(c);
                    c = chars[i];
                    num = 1;
                }
            }                
        }
        return cur.toString();
    }
}
```

### 39. 组合总和

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：

所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 

**示例 1：**

```
输入：candidates = [2,3,6,7], target = 7,
所求解集为：
[
  [7],
  [2,2,3]
]
```

**示例 2：**

```
输入：candidates = [2,3,5], target = 8,
所求解集为：
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

链接：https://leetcode-cn.com/problems/combination-sum

#### 解法一：回溯法

组合问题是回溯法的经典题型了。

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return result;
        }
        Arrays.sort(candidates);
        List<Integer> path = new ArrayList<>();
        backTracking(candidates, 0, target, 0, path);
        return result;
    }
    //prevSum保存上一次得到的总和，i表示现在进行的是以nums[i]为开头的组合
    private void backTracking(int[] nums, int prevSum, int target, int i, List<Integer> path) {
        if(prevSum == target) {
            result.add(new ArrayList<>(path));
            return;
        }
        for(int m = i; m < nums.length; m++) {
            if(nums[m] > target - prevSum) {
                break;
            }
            prevSum += nums[m];
            path.add(nums[m]);            
            backTracking(nums, prevSum, target, m, path);
            path.remove(path.size() - 1);
            prevSum -= nums[m];
        }
    }
}
```

#### 解法二：动态规划

这题能用动态规划我是真没想到。

意思就是题目让求和为target的集合，那就从和为0开始求。用一个list起名为opt。opt[0]表示和为0的所有情况组合，然后求opt[1]、opt[2]、……、opt[target]。

遍历数组中的元素，会遇到三种情况：

- 与target相等：此时直接把该元素添加入即可
- 大于target：这代表从这个元素开始后面的所有元素都不会再参与到opt[target]的组成，直接break。
- 小于target：找到opt[target-nums[i]]，将这里面的每个列表都加入nums[i]再放入到opt[target]

此时会出现一个问题，就是会重复添加元素。例如对于数组{2,3,5,6}，求opt[5]的时候，先遍历2，得到opt[3] = {3}，加入2放入opt，然后遍历3，得到opt[2] = {2}，加入3放入opt。这样就重复了。

我采用的去重方法就是判断获得到的集合的最后一个元素是否大于当前遍历元素，如果大于就不放入。保证集合里的元素永远是单调增的，不会有重复。

```java
class Solution {
    
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<List<Integer>>();
        }        
        Arrays.sort(candidates);
        if (candidates[0] > target) {
            return new ArrayList<List<Integer>>();
        }

        List<List<List<Integer>>> opt = new ArrayList<>();
        for(int sum = 0; sum <= target; sum++) {
            List<List<Integer>> ans = new ArrayList<>();
            for(int j = 0; j < candidates.length; j++) {
                if(candidates[j] > sum) break;
                if(candidates[j] == sum) {
                    List<Integer> temp = new ArrayList<>();
                    temp.add(sum);
                    ans.add(temp);
                }else {
                    List<List<Integer>> pre = opt.get(sum - candidates[j]);
                    for(int i = 0; i < pre.size(); i++) {
                        if(pre.get(i).size() > 0 && pre.get(i).get(pre.get(i).size() - 1) > candidates[j]) continue;//去重
                        List<Integer> temp = new ArrayList<>(pre.get(i));
                        temp.add(candidates[j]);
                        ans.add(temp);
                    }
                }
            }
            opt.add(ans);
        }
        return opt.get(target);
    }
}
```

看到有一种[去重方法](https://leetcode.wang/leetCode-39-Combination-Sum.html)还挺巧妙的。上面的写法是用了两层 for 循环，分别对 opt 和 nums 进行遍历。可以把两层循环颠倒，外层遍历 nums，内层遍历 opt。

考虑 nums [ 0 ]，求出 opt [ 0 ]，求出 opt [ 1 ]，求出 opt [ 2 ]，求出 opt [ 3 ] ... 求出 opt [ 7 ]。

考虑 nums [ 1 ]，求出 opt [ 0 ]，求出 opt [ 1 ]，求出 opt [ 2 ]，求出 opt [ 3 ] ... 求出 opt [ 7 ]。

考虑 nums [ 2 ]，求出 opt [ 0 ]，求出 opt [ 1 ]，求出 opt [ 2 ]，求出 opt [ 3 ] ... 求出 opt [ 7 ]。

考虑 nums [ 3 ]，求出 opt [ 0 ]，求出 opt [ 1 ]，求出 opt [ 2 ]，求出 opt [ 3 ] ... 求出 opt [ 7 ]。

```java
class Solution {
    
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<List<Integer>>();
        }        
        Arrays.sort(candidates);
        if (candidates[0] > target) {
            return new ArrayList<List<Integer>>();
        }
        List<List<List<Integer>>> opt = new ArrayList<>();
        //初始化opt
        for(int i = 0; i <= target; i++) {
            List<List<Integer>> ans_sum = new ArrayList<>();
            opt.add(ans_sum);
        }
        for(int i = 0; i < candidates.length; i++) {
            for(int sum = 0; sum <= target; sum++) {
                if(sum < candidates[i]) continue;
                List<List<Integer>> ans_sum = opt.get(sum);
                if(sum == candidates[i]) {
                    List<Integer> temp = new ArrayList<>();
                    temp.add(sum);
                    ans_sum.add(temp);
                }else {
                    List<List<Integer>> ans_sub = opt.get(sum - candidates[i]);
                    for(int j = 0; j < ans_sub.size(); j++) {
                        ArrayList<Integer> temp = new ArrayList<Integer>(ans_sub.get(j));
                        temp.add(candidates[i]);
                        ans_sum.add(temp);
                    }
                }
            }
        }
        return opt.get(target);
    }
}
```

### 40. 组合总和II

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

**说明：**

- 所有数字（包括目标数）都是正整数。
- 解集不能包含重复的组合。

 **示例 1:**

```
输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```

**示例 2:**

```
输入: candidates = [2,5,2,1,2], target = 5,
所求解集为:
[
  [1,2,2],
  [5]
]
```

链接：https://leetcode-cn.com/problems/combination-sum-ii

#### 解法：回溯法

就是在上一题的基础上做一些修改。遇上一题最大的区别在于：数组中会有重复元素， 组合中不允许重复利用同一元素。

为了要保证不利用同一元素，这代表，每次调用递归函数时，传入的下一个for循环开始点是当前位置+1。除此以外，在本层的for循环中要注意，遇到值相同的元素要跳过。

以{2,2,2,4}为例，要求target = 8，首先从第一个2开始，调用第二层递归从第二个2开始，进入第三层递归从第三个2开始，第四层从4开始，发现总和大于8，不满足条件，弹栈。回到第三层递归遍历到4，满足target=8，将该组合{2,2,4}加入到结果中。再弹栈，回到第二层递归，这时按道理应该遍历到了第三个2，但是由于第二个2在第二层递归中已经利用过，就直接跳过……弹栈到第一层的时候，本该继续遍历第二个2，但是按照上面分析的，该层已经遍历过一个2，后面两个2就不应该再在本层使用了，所以跳过。这样最终得到的结果只有{2,2,4}

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    int length;
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        length = candidates.length;
        Arrays.sort(candidates);
        List<Integer> path = new ArrayList<>();
        backTracking(candidates, target, 0, 0, path);
        return result;
    }
    private void backTracking(int[] nums, int target, int prevSum, int m, List<Integer> path) {
        if(prevSum == target) {
            result.add(new ArrayList<>(path));
            return;
        }
        for(int i = m; i < length; i++) {
            if(nums[i] > target - prevSum) break;
            //表明nums[i]和nums[i - 1]相等，且nums[i - 1]不是这一层函数里遍历经过的。
            if(i > m && nums[i] == nums[i - 1]) continue;
            prevSum += nums[i];
            path.add(nums[i]);
            backTracking(nums, target, prevSum, i + 1, path);            
            prevSum -= path.remove(path.size() - 1);
        }
    }
}
```

### 45. 跳跃游戏II

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。

**示例 1:**

```
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

**示例 2:**

```
输入: [2,3,0,1,4]
输出: 2
```

链接：https://leetcode-cn.com/problems/jump-game-ii

#### 解法一：贪心算法

其实就是找当前位置所能达到的下一跳的最远位置。

以[2,3,1,1,4,2,1]为例，在位置0，能达到的最远位置是2。

在位置1，能达到的最远位置是4，但是这个时候还不能跳！因为上一次能达到的最远位置是位置2，我们无法确定是否可以通过位置2达到一个更远的位置。所以还要看位置2，它能达到的最远位置是3，不如4大。所以确定了要先从0跳到1。

当前可以达到的最远位置是4。遍历到位置3，发现位置3最远到位置4，遍历到位置4，最远可达位置是8，超过了数组长度，比从位置3开始跳能到达的范围更广一些。且位置4恰好是当前所能达到的最远位置，更新从1跳到4，然后从4即可跳到末尾。

```java
class Solution {
    public int jump(int[] nums) {
        int far = 0;
        int nextfar = nums[0];
        int step = 0;
        for(int i = 0; i <= far && far < nums.length - 1; i++) {
            //求当前范围内可以达到的下一跳最远位置
            nextfar = Math.max(nextfar, i + nums[i]);
            if(i == far) {
                //已经到达了上一跳所能到达的最远位置
                //要再跳一次
                //并且要更新最远位置
                far = nextfar;
                step++;
            }
        }
        return step;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法二：动态规划

首先明确dp数组的含义，dp[i]表示当前位置i要走到末尾需要的最少步数。

从后往前遍历数组。

```java
class Solution {
    public int jump(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp,nums.length-1);
        //dp[i]表示当前位置i要走到末尾最少需要的步数。
        for(int i = nums.length - 2; i >= 0; i--) {
            for(int step = 1; step <= nums[i]; step++) {
                if(i+step >= nums.length - 1) {
                    dp[i] = 1;
                    break;
                }else {
                    dp[i] = Math.min(dp[i],1+dp[i+step]);
                }                
            }
        }
        return dp[0];
    }
}
```

时间复杂度：最坏的情况是O（n^2^）

空间复杂度：O（n）

### 46. 全排列

给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**示例 2：**

```
输入：nums = [0,1]
输出：[[0,1],[1,0]]
```

**示例 3：**

```
输入：nums = [1]
输出：[[1]]
```

链接：https://leetcode-cn.com/problems/permutations

#### 解法一：回溯法

用一个布尔数组记录该数是否已经被列入排列中，如果不是就可以将该数加入list，然后调用下一层的递归。

![](https://pic.leetcode-cn.com/0bf18f9b86a2542d1f6aa8db6cc45475fce5aa329a07ca02a9357c2ead81eec1-image.png)

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    boolean[] marked;
    public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        if(len == 0) return result;
        marked = new boolean[len];
        List<Integer> path = new ArrayList<>();
        dfs(nums, len, 0, path);
        return result;
    }
    private void dfs(int[] nums, int len, int depth, List<Integer> path) {
        if(depth == len) {
            result.add(new ArrayList<>(path));
            return;
        }
        for(int i = 0; i < len; i++) {
            if(marked[i] == false) {
                marked[i] = true;
                path.add(nums[i]);
                dfs(nums, len, depth+1, path);
                marked[i] = false;
                path.remove(path.size() - 1);
            }
        }
    }
}
```

#### 解法二：回溯法

排列，顾名思义，就是把一组数放在不同的位置上，这个过程也可以看成是不停地交换位置。回溯递归函数的每一层都有一个for循环，从指定的位置开始遍历，每一个数都与指定位置发生一次交换。

```java
class Solution {
    //123456
    //每层递归每个位置都与一个指定位置发生交换
    //第一层每个位置分别与1交换
    //第二层每个位置分别和2交换
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        if(len == 0) return result;     
        dfs(nums, len, 0);
        return result;
    }
    private void dfs(int[] nums, int len, int depth) {
        if(depth == len - 1) {
            List<Integer> path = new ArrayList<>();
            for(int num : nums) {
                path.add(num);
            }
            result.add(path);
            return;
        }
        for(int i = depth; i < len; i++) {
            swap(nums, i, depth);
            dfs(nums, len, depth + 1);
            swap(nums, i, depth);
        }
    }
    private void swap(int[] nums, int i, int j) {
        int ss = nums[i];
        nums[i] = nums[j];
        nums[j] = ss;
    }
}
```

### 47. 全排列II

给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。

**示例 1：**

```
输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]
```

**示例 2：**

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

链接：https://leetcode-cn.com/problems/permutations-ii

#### 解法一：回溯

这一题用46的解法一就可以了，但是注意要添加一些条件。

因为是包含着重复数字的，一般遇到重复数字都需要先排序，这样便于跳过那些重复的数字。排序后进入回溯递归函数，以示例1作例子，先求以第一个数字打头的排列可以得到112和121，然后求第二个数字打头的排列，应当跳过。

那么推广到一般情况遇到什么样的数字应该跳过呢？首先可以确定的是该数字一定和前一个数字相等。但是并不是所有相等数字都要跳过，比如在上面例子中，如果在求以第一个数字打头的排列中，遍历到下一个数字1发现和前一个数字相等就跳过的话，求出的只有12了。所以还要添加一个条件即，前一个数字的marked状态是false。这代表，以前面一个数开头的所有排列已经求过了，它的状态已经被还原了。所以现在再求当前这个与前面数字相等的数字的所有排列的话就是重复了。

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    boolean[] marked;
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        int len = nums.length;
        marked = new boolean[len];
        List<Integer> path = new ArrayList<>();
        dfs(nums, len, 0, path);
        return result;
    } 
    private void dfs(int[] nums, int len, int depth, List<Integer> path) {        
        if(depth == len) {
            result.add(new ArrayList<>(path));
            return;
        }
        for(int i = 0; i < len; i++) {
            if(i > 0 && nums[i] == nums[i - 1] && !marked[i - 1]) continue;

            if(marked[i] == false) {
                marked[i] = true;
                path.add(nums[i]);
                dfs(nums, len, depth+1, path);
                marked[i] = false;
                path.remove(path.size() - 1);
            }
        }
    }
}
```

#### 解法二：回溯法

用和46题解法二相同的思路，本来想的是在depth层遇到了和前一个数字相等的情况就跳过，但是提交后发现还是会有重复情况。贴上错误代码。

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        dfs(nums, 0);
        return result;
    } 
    private void dfs(int[] nums, int depth) {        
        if(depth == nums.length - 1) {
            List<Integer> path = new ArrayList<>();
            for(int num : nums) {
                path.add(num);
            }
            result.add(path);
            return;
        }
        for(int i = depth; i < nums.length; i++) {
            if(i > depth && nums[i] == nums[i-1]) continue;
            swap(nums,i,depth);
            dfs(nums,depth+1);
            swap(nums,i,depth);
        }        
    }
    private void swap(int[] nums, int i, int j) {
        int ss = nums[i];
        nums[i] = nums[j];
        nums[j] = ss;
    }
}
```

在测试样例[0,0,0,1,9]上就过不去了，会重复出现09010。这是因为在第二层遍历到末尾数字和第二个数字进行交换后得到了09010，然后从第五层回退到第四层，第五个数字和第四个数字进行交换得到了09001，然后从第五层回退到第四层再退到第三层，得到了09100，然后从第五层又退到第三层，第四个数字要和第三个数字交换得到了09010。

发生这种错误是因为经过了位置交换后破坏了原来的有序。

所以还是要用最原始的方法，用一个哈希表把所有的排列都装进去，如果哈希表里已经有了这种排列就跳过。之前的去重方法也可以保留，当做一种剪枝思想。

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    HashSet<List<Integer>> group = new HashSet<>();
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        dfs(nums, 0);
        return result;
    } 
    private void dfs(int[] nums, int depth) {        
        if(depth == nums.length - 1) {
            List<Integer> path = new ArrayList<>();
            for(int num : nums) {
                path.add(num);
            }
            if(!group.contains(path)){
                result.add(path);
                group.add(path);
            }
            return;
        }
        for(int i = depth; i < nums.length; i++) {
            if(i > depth && nums[i] == nums[i-1]) continue;
            swap(nums,i,depth);
            dfs(nums,depth+1);
            swap(nums,i,depth);
        }        
    }
    private void swap(int[] nums, int i, int j) {
        int ss = nums[i];
        nums[i] = nums[j];
        nums[j] = ss;
    }
}
```

### 50. Pow(x,n)

实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，x^n^）。

**示例 1：**

```
输入：x = 2.00000, n = 10
输出：1024.00000
```

**示例 2：**

```
输入：x = 2.10000, n = 3
输出：9.26100
```

**示例 3：**

```
输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25
```

链接：https://leetcode-cn.com/problems/powx-n

#### 解法一：递归二分

如果按照正常思路x相乘n次，遇到n特别大的情况是一定会超时的，那么可以通过把n/2。例如若n = 8，其实就是两个x相乘后的结果，做平方变成四次，再平方变成8次。如果遇到n = 9，就可以先两个x相乘后的结果，做平方变成4次（4 = 9/2），再平方乘上原先的x即可得到9次方。

```java
class Solution {
    public double myPow(double x, int n) {
        if(n == 0) return 1;
        if(n == 1) return x;
        if(n == -1) return 1/x;
        double p1 = myPow(x,n/2);
        if(n/2 == n - n/2) {
            //n是偶数
            return p1 * p1;
        }else if(n/2 < n - n/2){
            //n是正奇数
            return p1 * p1 * x;
        }else {
            //n是负奇数
            return p1 * p1 / x;
        }
    }
}
```

时间复杂度：O（log n）

空间复杂度：O（1）

n/2可以写成n>>1，位操作会更快一点。

#### 解法二：迭代

参考[题解](https://leetcode-cn.com/problems/powx-n/solution/50-powx-n-kuai-su-mi-qing-xi-tu-jie-by-jyd/)

以x 的 10 次方举例。10 的 2 进制是 1010，然后用 2 进制转 10 进制的方法把它展成 2 的幂次的和。
$$
x^{10}=x^{(1010)_2}=x^{2^3}*x^{2^1}
$$
![](https://windliang.oss-cn-beijing.aliyuncs.com/50_2.jpg)

即对应1的项累乘。x不断更新。

当遇到n为最小负数 -2^31^，取相反数的话，按照计算机的规则，依旧是-2^31^，这种情况需要单独讨论。将-2^31^拆解成-2^31^+1和-1即可。

```java
class Solution {
    public double myPow(double x, int n) {
        if(n == 0) return 1;
        if(n == -2147483648) {
            //特殊情况
            return myPow(x, -2147483647) * (1 / x);
        }

        if(n < 0) {
            n = -n;
            x = 1/x;
        }        
        double res = 1;
        while(n > 0) {
            if((n & 1) == 1) res *= x;
            n >>= 1;
            x = x * x;
        }
        return res;
    }
}
```

时间复杂度：O（log n）

空间复杂度：O（1）

### 51. N皇后

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

 皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。

**示例 1：**

<img src="https://assets.leetcode.com/uploads/2020/11/13/queens.jpg" style="zoom:80%;" />

```
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。
```

**示例 2：**

```
输入：n = 1
输出：[["Q"]]
```

链接：https://leetcode-cn.com/problems/n-queens

#### 解法：回溯法

这题也是很经典的回溯算法题了。

因为每一行只能放一个棋子，所以可以用一个数组nums[i]代表的是i行的棋子放在第几列。

在第depth层，求的是nums[depth]的值，把从1到n的每个值都试一遍，如果值被用过了就跳过，如果没用过，还要检查前面每一行棋子摆放的位置是否会这一行棋子位置构成斜线。如果所有数字试过了都不满足条件，这代表前面棋子摆放的位置有问题，那么就不进行递归了，直接return。如果有位置满足要求，那么就进入depth+1层的递归调用。

```java
class Solution {
    int[] nums;
    boolean[] marked;
    int all;
    List<List<String>> result = new ArrayList<>();
    public List<List<String>> solveNQueens(int n) {
        nums = new int[n];
        marked = new boolean[n];
        all=n;
        for(int i = 0; i < n; i++) {
            nums[i] = i;
        }
        backTracking(0);
        return result;
    }
    private void backTracking(int depth) {
        if(depth==all){
            List<String> p = new ArrayList<>();            
            for(int i = 0; i < all; i++) {
                StringBuilder path = new StringBuilder();
                for(int j = 0; j < all; j++) {
                    if(j==nums[i]) {
                        path.append('Q');
                    }else {
                        path.append('.');
                    }
                }
                p.add(path.toString());
            }
            result.add(p);
            return;
        }
        for(int i = 0; i < all; i++) {                       
            if(!marked[i]) {//该数字是否被取值过
                boolean flag = true; 
                if(depth > 0) {
                    if(Math.abs(i-nums[depth-1]) > 1) {//判断是否在同一列或者是相邻两行的斜线上
                        for(int j = depth-1; j >= 0; j--) {
                            if(Math.abs(i-nums[j]) == depth - j) {
                                //判断是否在跨行斜线上，是的话，则不满足条件不进行递归
                                flag = false;
                                break;
                            }
                        }
                    }else {//如果在同一列或者是相邻两行的斜线上，则不进行递归
                        flag = false;
                    }
                }
                if(flag) {
                    nums[depth] = i;
                    marked[i] = true;
                    backTracking(depth+1);
                    marked[i] = false;
                }
            } 
        }
    }
}
```

### 52. N皇后II

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案数量。

**示例 1：**

<img src="https://assets.leetcode.com/uploads/2020/11/13/queens.jpg" style="zoom:80%;" />

```
输入：n = 4
输出：2
```

**示例 2：**

```
输入：n = 1
输出：1
```

链接：https://leetcode-cn.com/problems/n-queens-ii/

#### 解法一：回溯法

在上一题的基础上直接做修改即可得到

```java
class Solution {
    int count = 0;
    int all;
    boolean[] marked;
    int[] nums;
    public int totalNQueens(int n) {
        marked = new boolean[n];
        nums = new int[n];
        all = n;
        backTracking(0);
        return count;
    }
    private void backTracking(int depth) {
        if(depth==all){
            count++;
            return;
        }
        for(int i = 0; i < all; i++) {                       
            if(!marked[i]) {//该数字是否被取值过
                boolean flag = true; 
                if(depth > 0) {
                    if(Math.abs(i-nums[depth-1]) > 1) {//判断是否在同一列或者是相邻两行的斜线上
                        for(int j = depth-1; j >= 0; j--) {
                            if(Math.abs(i-nums[j]) == depth - j) {
                                //判断是否在跨行斜线上，是的话，则不满足条件不进行递归
                                flag = false;
                                break;
                            }
                        }
                    }else {//如果在同一列或者是相邻两行的斜线上，则不进行递归
                        flag = false;
                    }
                }
                if(flag) {
                    nums[depth] = i;
                    marked[i] = true;
                    backTracking(depth+1);
                    marked[i] = false;
                }
            } 
        }
    }
}
```

#### 解法二：回溯法

同样是回溯法，稍稍不同的是在对角线的判断上。

<img src="https://windliang.oss-cn-beijing.aliyuncs.com/52_2.jpg" style="zoom:80%;" />

观察上面的图，可以得到对于同一条主对角线，row - col 的值是相等的。对于同一条副对角线，row + col 的值是相等的。所以主副对角线是否有棋子可以分别用两个布尔型数组来保存。

```java
class Solution {
    int count = 0;
    int all;
    boolean[] marked;
    boolean[] zhu;
    boolean[] fu;
    public int totalNQueens(int n) {
        marked = new boolean[n];
        zhu = new boolean[2*n];
        fu = new boolean[2*n];
        all = n;
        backTracking(0);
        return count;
    }
    private void backTracking(int depth) {
        if(depth==all){
            count++;
            return;
        }
        for(int i = 0; i < all; i++) {   
            int d1 = depth - i + all, d2 = depth + i;                    
            if(!marked[i] && !zhu[d1] && !fu[d2]) {//该数字是否被取值过、是否在主副对角线上有棋子
                zhu[d1] = true;
                fu[d2] = true;
                marked[i] = true;
                backTracking(depth+1);
                marked[i] = false;
                zhu[d1] = false;
                fu[d2] = false;
            } 
        }
    }
}
```

### 53. 最大子序和

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**示例 1：**

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

**示例 2：**

```
输入：nums = [1]
输出：1
```

**示例 3：**

```
输入：nums = [0]
输出：0
```

**示例 4：**

```
输入：nums = [-1]
输出：-1
```

**示例 5：**

```
输入：nums = [-100000]
输出：-100000
```

链接：https://leetcode-cn.com/problems/maximum-subarray

#### 解法一：贪心

思路是只要发现前一子序列和prev是负的就以当前位置重新开始计算，如果不是负的，就加上当前位置的。每一次更新prev也要同步更新res。

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int len = nums.length;
        if(len == 1) return nums[0];
        int prev = Integer.MIN_VALUE, res = Integer.MIN_VALUE;
        for(int num : nums) {
            if(prev < 0) {
                prev = num;
            }else {
                prev += num;
            }
            res = Math.max(res, prev);
        }
        return res;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法二：动态规划

其实和上面的解法本质一样。

dp[i] 代表的是以 nums[i] 为结尾的子序列所能得到的最大和。即如果dp[i-1]是负的，那么dp[i]最大肯定是抛弃之前的子序列，只包含nums[i]。

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int len = nums.length;
        int[] dp = new int[len];
        int res = Integer.MIN_VALUE;
        for(int i = 0; i < len; i++) {
            dp[i] = (i > 0 && dp[i-1] > 0) ? dp[i-1] + nums[i] : nums[i];
            res = Math.max(res,dp[i]);
        }
        return res;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

### 54. 螺旋矩阵

给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。 

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg)

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2020/11/13/spiral.jpg)

```
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

链接：https://leetcode-cn.com/problems/spiral-matrix



观察一下整个过程，先是列数增加，增到最大值之后转向；变为行数增加，增到最大值后转向；变为列减少，减到最小值时转向；变为行数减少，减到最小值后再转到列数增加。

在这过程中，行列能达到的最大最小值也需要更新，可以确定的是列增操作时，一定是在最小行，行增时在最大列，列减时在最大行，行减时在最小列。每次转向前把自己所在的那个边界值做修改即可。

以示例2为例，到达4时，需要转向，并且要将行最小值0加一变成1。到达12时，要把列最大值减1改为2。

```java
class Solution {    
    public List<Integer> spiralOrder(int[][] matrix) {
        //分别表示行最小值，行最大值，列最小值，列最大值
        int rowD, rowU, colD, colU;
        rowD = colD = 0;
        rowU = matrix.length - 1;
        colU = matrix[0].length - 1;
        
        List<Integer> result = new ArrayList<>();

        int row = 0, col = 0;//记录坐标
        int N = matrix.length * matrix[0].length;
        //rowAsc表示行增, rowDesc表示行减, colAsc表示列增, colDesc表示列减
        boolean rowAsc = false, rowDesc = false, colAsc = false, colDesc = false;
        if(colU == colD) {
            rowAsc = true;
        }else {
            colAsc = true;
        }
        while(N-- > 0) {
            result.add(matrix[row][col]);
            if(colAsc) {
                col++;
                if(col == colU) {
                    rowAsc = true;
                    colAsc = false;
                    rowD++;
                }
            }else if(rowAsc) {
                row++;
                if(row == rowU) {
                    colDesc = true;
                    rowAsc = false;
                    colU--;
                }
            }else if(colDesc) {
                col--;
                if(col == colD) {
                    rowDesc = true;
                    colDesc = false;
                    rowU--;
                }
            }else if(rowDesc) {
                row--;
                if(row == rowD) {
                    colAsc = true;
                    rowDesc = false;
                    colD++;
                }
            }
        }
        return result;
    }
}
```

### 55. 跳跃游戏

给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

**示例 1：**

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

**示例 2：**

```
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```

链接：https://leetcode-cn.com/problems/jump-game

感觉这题还是挺简单的。

随便给几种写法。

下面这种写法和前面45题是一样的本质。

```java
class Solution {
    public boolean canJump(int[] nums) {
        int far = nums[0];
        int nextfar = nums[0];
        for(int i = 1; i < nums.length; i++) {
            if(i > far) {
                return false;
            }

            nextfar = Math.max(nextfar, i+nums[i]);
            
            if(i == far) {
                far = nextfar;
            }
        }
        return true;
    }
}
```

下面这个写法比较直击问题本质。

```java
class Solution {
    public boolean canJump(int[] nums) {
        int far = 0;
        for(int i = 0; i < nums.length; i++) {
            if(i > far) return false;
            far = Math.max(far, i + nums[i]);
            if(far >= nums.length - 1) return true;
        }
        return true;
    }
}
```

### 56. 合并区间

以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [start~i~, end~i~] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

**示例 1：**

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

**示例 2：**

```
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
```

链接：https://leetcode-cn.com/problems/merge-intervals

要看两个区间是否重叠，首先要搞清楚这两个区间谁的起始点比较靠后，然后看起始点较大的数值是否大于起始点较小区间的末尾。如果大于，那么不重叠，如果小于等于，就会重叠。

所以这题首先要将所有的区间以start~i~排序，然后再看他们是否有首尾相接的情况。

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if(intervals.length == 1) return intervals;
        List<int[]> list= new ArrayList<>();
        //以starti排序
        Arrays.sort(intervals,  (a, b) -> a[0] - b[0]);
        int start = intervals[0][0], end = intervals[0][1];
        for(int i = 1; i < intervals.length; i++) {
            if(intervals[i][0] <= end) {
                //如果当前区间的起始点小于之前记录下的区间结尾
                //更新end
                //切记这里不能直接end = intervals[i][1]，因为很有可能当前遍历的区间包含在之前的区间里
                end = Math.max(end,intervals[i][1]);
            }else {
                list.add(new int[]{start,end});
                start = intervals[i][0];
                end = intervals[i][1];
            }
        }
        //最后一次修改的区间要加入到list中。
        list.add(new int[]{start,end}); 
        //最后将list转换为数组
        int[][] res = new int[list.size()][2];
        int i = 0;
        for(int[] l : list) {
            res[i] = l;
            i++;
        }
        return res;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

### 57. 插入区间

给你一个 无重叠的 ，按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

**示例 1：**

```
输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
输出：[[1,5],[6,9]]
```

**示例 2：**

```
输入：intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出：[[1,2],[3,10],[12,16]]
解释：这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。
```

**示例 3：**

```
输入：intervals = [], newInterval = [5,7]
输出：[[5,7]]
```

**示例 4：**

```
输入：intervals = [[1,5]], newInterval = [2,3]
输出：[[1,5]]
```

**示例 5：**

```
输入：intervals = [[1,5]], newInterval = [2,7]
输出：[[1,7]]
```

链接：https://leetcode-cn.com/problems/insert-interval

#### 解法一

利用56题的代码成果，先把新的数组加入到一个新的大数组中，然后对于这个新的大数组调用56题的代码。

```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        int[][] inter = new int[intervals.length+1][2];
        boolean add = false; //标记是否已经将newInterval加入新数组
        for(int i = 0; i < intervals.length + 1; i++) {
            if(!add) {
                if(i == intervals.length) {
                    //如果已经超过了原数组边界，直接将newInterval加入
                    inter[i] = newInterval;
                    add = true;
                }else {
                    if(intervals[i][0] < newInterval[0]) {
                        //如果当前intervals[i]的起点小于newInterval的起点，加入intervals[i]
                        inter[i] = intervals[i];
                    }else {
                        //将interval加入，注意要将add修改状态
                        inter[i] = newInterval;
                        add = true;
                    }
                }
            }else {
                //已经加入了newInterval后，inter数组和intervals数组的索引对应关系要变
                inter[i] = intervals[i-1];
            }
        }
        return merge(inter);
    }
    private int[][] merge(int[][] intervals) {
        if(intervals.length == 1) return intervals;
        List<int[]> list= new ArrayList<>();
        //以starti排序
        Arrays.sort(intervals,  (a, b) -> a[0] - b[0]);
        int start = intervals[0][0], end = intervals[0][1];
        for(int i = 1; i < intervals.length; i++) {
            if(intervals[i][0] <= end) {
                //如果当前区间的起始点小于之前记录下的区间结尾
                //更新end
                //切记这里不能直接end = intervals[i][1]，因为很有可能当前遍历的区间包含在之前的区间里
                end = Math.max(end,intervals[i][1]);
            }else {
                list.add(new int[]{start,end});
                start = intervals[i][0];
                end = intervals[i][1];
            }
        }
        //最后一次修改的区间要加入到list中。
        list.add(new int[]{start,end}); 
        //最后将list转换为数组
        return list.toArray(new int[list.size()][]);
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

执行时间比较慢的，需要4 ms，只击败了18.05%

#### 解法二

其实重新生成一个矩阵的过程也非常耗时，那么如何在不重新创建矩阵的情况下进行合并呢？

我的想法如下。需要加入很多判断条件。

```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        if(intervals.length == 0) {
            int[][] res = new int[1][2];
            res[0] = newInterval;
            return res;
        }
        List<int[]> list= new ArrayList<>();
        int start = newInterval[0], end = newInterval[1];
        //addstart表示start已经最终确定，addend表示end已经最终确定。
        boolean addstart = false, addend = false;
        for(int i = 0; i < intervals.length; i++) {
            if(!addstart) {
                if(intervals[i][1] < newInterval[0]) {
                    //如果当前小区间的结尾小于newInterval的起始
                    //代表当前区间在newInterval的前面，类似于当前区间为[1,2]而newInterval是[4,5]
                    //可以直接把当前区间加入list中
                    list.add(intervals[i]);
                }else {                    
                    if(intervals[i][0] > newInterval[1]) {
                        //如果当前区间的起始大于newInterval的结尾
                        //类似于当前区间为[3,5]而newInterval是[1,2]，两个区间无重叠
                        //直接将两个区间独立加入list即可
                        list.add(newInterval);
                        list.add(intervals[i]);
                        //末尾已经确定过了
                        addend = true;
                    }else {
                        //当前区间和newInterval有重叠
                        //start取较小的值
                        start = Math.min(intervals[i][0], newInterval[0]);
                        //如果当前区间的结尾大于等于newInterval的结尾，代表合并区间的结尾也已经可以确定了
                        if(intervals[i][1] >= newInterval[1]) {
                            addend = true;//修改状态
                            end = intervals[i][1];
                            list.add(new int[] {start,end});
                        }else {
                            //end还不可以被最终确定
                            end = newInterval[1];
                        }
                    }
                    //start已经被确定
                    addstart = true;
                }
            }else if(!addend) {
                if(end < intervals[i][0]) {
                    //end如果小于当前区间
                    //代表与当前区间无重叠
                    //直接将[start，end]和当前区间分别加入即可
                    addend = true;
                    list.add(new int[] {start,end});
                    list.add(intervals[i]);
                }else if(end <= intervals[i][1]) {
                    //如果当前区间的结尾大于end
                    //那么end就是当前区间的结尾
                    addend = true;
                    end = intervals[i][1];
                    list.add(new int[] {start,end});
                }
            }else {
                //start和end都已经确定，不需要再做其他工作
                //直接把数组加入list即可
                list.add(intervals[i]);
            }
            if(i == intervals.length - 1 && !addend) list.add(new int[] {start,end});
        }
        //最后将list转换为数组
        return list.toArray(new int[list.size()][]);
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

执行时间1 ms，击败了99%

这么多判断条件还是很绕人的，看到了一种非常简洁明了的写法。着实妙啊！

首先第一个循环，在没遇到比newInterval[0]大的末尾点前先将所有的数组按照顺序加入。

跳出第一个while循环后，进入第二个while循环。这里将newInterval作为装载最终呈现区间的容器，即如果newInterval与一些区间有重合，那么最终newInterval存放的是合并后的区间，如果不重合那么存放的是自身。如果newInterval的末尾点也已经确定好了就跳出循环。

第三个循环是将intervals矩阵中还没遍历到的加入到list中。

这种写法几个比较妙的点：1. 不同步骤分别用独立循环；2. 将newInterval作为一个可修改的容器。

```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        ArrayList<int[]> list = new ArrayList<>();
        int len = intervals.length;
        int i = 0;
        while(i < len && intervals[i][1] < newInterval[0]) {
            list.add(intervals[i]);
            i++;
        }
        while(i < len && intervals[i][0] <= newInterval[1] ) {
            newInterval[0] = Math.min(intervals[i][0], newInterval[0]);
            newInterval[1] = Math.max(intervals[i][1], newInterval[1]);
            i++;
        }
        list.add(newInterval);
        while(i < len) {
            list.add(intervals[i]);
            i++;
        }
        return list.toArray(new int[list.size()][]);
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

执行时间1 ms，击败了99%

### 58. 最后一个单词的长度

给你一个字符串 s，由若干单词组成，单词之间用空格隔开。返回字符串中最后一个单词的长度。如果不存在最后一个单词，请返回 0 。

单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。

**示例 1：**

```
输入：s = "Hello World"
输出：5
```

**示例 2：**

```
输入：s = " "
输出：0
```

链接：https://leetcode-cn.com/problems/length-of-last-word



这题的一个易错点就在于：字符串可能是以多个空格为结尾的。

#### 解法一：

我的写法是，用两个变量存储。一个记录当前单词的长度，一个记录上次遇到空格之前的有效单词长度。

```java
class Solution {
    public int lengthOfLastWord(String s) {
        int cur = 0; //记录当前单词的长度
        int prev = 0; //记录空格前一个单词的长度
        char[] chars = s.toCharArray();
        for(int i = 0; i < chars.length; i++) {
            if(chars[i] == ' ') {
                //如果cur已经是0代表当前空格前面没有有效的单词
                if(cur > 0) {
                    //如果cur大于0，需要更新prev，并将cur重新置位0                    
                    prev = cur; 
                    cur = 0;  
                }                  
            }else {
                cur++;
            }
        }
        return cur == 0 ? prev : cur;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

#### 解法二：

看到一种解法，挺简单的。从后往前遍历减小index指针，遇到非空格就停止。然后再继续从index往前遍历，开始计算长度，遇到空格就跳出。

```java
class Solution {
    public int lengthOfLastWord(String s) {
        int count = 0;
        int index = s.length() - 1;
        char[] chars = s.toCharArray();
        while(index >= 0 && chars[index] == ' ') {
            index--;
        }
        while(index >= 0) {
            if(chars[index] == ' ') break;
            count++;
            index--;
        }
        return count;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

这种写法的执行时间更快一些。妙就妙在从后往前的思想，而且也是拆分成了两个循环。上一题的最后一种写法也是妙在有拆成几个循环的做法。

### 59. 螺旋矩阵II

给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。 

**示例 1：**

```
输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]
```


**示例 2：**

```
输入：n = 1
输出：[[1]]
```

链接：https://leetcode-cn.com/problems/spiral-matrix-ii

#### 解法一：

这一题和上面54题的思路都是一样的，套一下就可以了。

```java
class Solution {
    public int[][] generateMatrix(int n) {
        int rowD, rowU, colD, colU;
        rowD = colD = 0;
        rowU = n - 1;
        colU = n - 1;
        
        int[][] matrix = new int[n][n];

        int row = 0, col = 0;//记录坐标
        int N = n*n;
        //rowAsc表示行增, rowDesc表示行减, colAsc表示列增, colDesc表示列减
        boolean rowAsc = false, rowDesc = false, colAsc = true, colDesc = false;
        
        for(int i = 1; i <= N; i++) {
            matrix[row][col] = i;
            if(colAsc) {
                col++;
                if(col == colU) {
                    rowAsc = true;
                    colAsc = false;
                    rowD++;
                }
            }else if(rowAsc) {
                row++;
                if(row == rowU) {
                    colDesc = true;
                    rowAsc = false;
                    colU--;
                }
            }else if(colDesc) {
                col--;
                if(col == colD) {
                    rowDesc = true;
                    colDesc = false;
                    rowU--;
                }
            }else if(rowDesc) {
                row--;
                if(row == rowD) {
                    colAsc = true;
                    rowDesc = false;
                    colD++;
                }
            }
        }
        return matrix;
    }
}
```

时间复杂度：O（N）

空间复杂度：O（N）

#### 解法二：

这题在一个多月前曾写过一次，当时对于拐点的处理方式是非常有趣的，虽然代码不是很好懂。

我通过查找规律的方式，发现了在除第四条边的其它三条边上转向时 格子内填的数 - 外面一圈的最大值 = （总层数 - 1 - 当前第几层*2） * 第几条边 + 1；在第四条边上要转向的点是 （格子内填的数 - 外面一圈的最大值）% （总层数 - 1 - 当前第几层*2） = 0。

当时也是莽，这么难找的规律给我硬掰扯出来了……

```java
class Solution {
    public int[][] generateMatrix(int n) {
        //bian代表是第几条边，layer代表是第几层，lout代表前一层的最大数
        int bian = 1, layer = 0, lout = 0;
        int x = 0, y = 0;
        int[][] res = new int[n][n];
        for(int i = 1; i <= n*n; i++) {
            res[x][y] = i;
            if(bian == 4 && (i - lout)%(n - 1 - layer*2) == 0) {//如果到了第四条边的最后一个数
                bian = 1;
                lout = i;
                layer++;//边置位1，lout更新，层数加一
            }else if((i - lout) == (n - 1 - layer*2) * bian + 1) {
                bian++;//在不是第四条边的拐点，边数增加
            }
            if(bian == 1) {
                y++;
            }else if(bian == 2) {
                x++;
            }else if(bian == 3) {
                y--;
            }else if(bian == 4) {
                x--;
            }
        }
        return res;
    }
}
```

时间复杂度：O（N）

空间复杂度：O（N）

### 60. 排序序列

给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：

"123"
"132"
"213"
"231"
"312"
"321"
给定 n 和 k，返回第 k 个排列。

**示例 1：**

```
输入：n = 3, k = 3
输出："213"
```

**示例 2：**

```
输入：n = 4, k = 9
输出："2314"
```

**示例 3：**

```
输入：n = 3, k = 1
输出："123"
```

链接：https://leetcode-cn.com/problems/permutation-sequence

#### 解法一：

首先观察以n=3为例的排列，共有6种排列方式。

先看第一个数字：当k=1和2时，以1开头；当k=3和4时，以2开头；当k=5和6时，以3开头。

再看第二个数字：以1开头还剩下2和3，k=1时为2，k=2时为3；以2开头还剩下1和3，k=3时为1，k=4时为3...

整个过程就是根据k落在哪一组取一个数字，再在剩下的数字里根据一个更新后的k值取下一个数字。

即以第一个数字可以将排列分成三组，每一组里有2*1=2种排列，所以选第一个数字的时候要看k落在哪一组。接着确定下一次需要找的是该大组中的第几小组。例如k=4，首先确定在第二大组里，然后减去第一大组的两种排列得到2，所以要找的是第二大组里的第二种排列。

```java
class Solution {
    //最终结果
    StringBuilder result = new StringBuilder();
    //存储按大小顺序排列的数字
    StringBuilder original = new StringBuilder();
    public String getPermutation(int n, int k) {
        for(int i = 1; i <= n; i++) {
            //将数字按顺序加入
            original.append(String.valueOf(i));
        }
        backTacking(n, k, 0);
        return result.toString();
    }
    private void backTacking(int n, int k) {
        if(n == 1) {
            result.append(original.charAt(0));
            return;
        }
        int comNum = 1;
        for(int i = 1; i < n; i++) {
            //计算大组里会有多少种排列
            comNum *= i;
        }
        int chu = k/comNum, yu = k%comNum;
        if(yu == 0) {
            //如果余数为0，索引要减1
            chu--;            
        }
        //取出数字
        result.append(original.charAt(chu));
        //把该数字从原始序列中删除
        original.deleteCharAt(chu);
        //得到下一次要搜索的是第几个排列
        yu = k - chu*comNum;
        backTacking(n - 1, yu);
    }    
}
```

时间复杂度：O（n）

空间复杂度：O（n）

#### 解法二：

直接利用之前写过的全排列代码

```java
class Solution {
    int count;
    boolean[] marked;
    StringBuilder path = new StringBuilder();
    public String getPermutation(int n, int k) {
        int[] nums = new int[n];
        marked = new boolean[n];
        for(int i = 1; i <= n; i++) {
            nums[i-1] = i;
        }
        count = k;
        backTacking(nums, 0);
        return path.toString();
    }
    private void backTacking(int[] nums, int depth) {
        if(depth == nums.length) {
            count--;
            return;
        }
        if(count == 0) return;
        for(int i = 0; i < nums.length; i++) {
            if(marked[i] == false) {
                marked[i] = true;
                path.append((char)(nums[i] + '0'));
                backTacking(nums, depth+1);
                if(count == 0) break;
                marked[i] = false;
                path.deleteCharAt(path.length() - 1);
            }
        }
    } 
}
```

这种方法虽然思路很简单，写起来也不复杂，但是非常耗时。

### 61. 旋转链表

给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/11/13/rotate1.jpg)

```
输入：head = [1,2,3,4,5], k = 2
输出：[4,5,1,2,3]
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2020/11/13/roate2.jpg)

```
输入：head = [0,1,2], k = 4
输出：[2,0,1]
```


链接：https://leetcode-cn.com/problems/rotate-list

#### 解法一：递归

一直递归调用，当遇到了尾结点，或者是还未将节点旋转完，就返回当前节点。如果已经完成了旋转，就返回头节点。

```java
class Solution {
    int K = 0, count = 0;
    ListNode H;
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null || k == 0) return head;
        K = k;
        H = head;
        return rotateHelp(head);
    }
    private ListNode rotateHelp(ListNode head) {
        //计算链表总节点数
        count++;
        if(head.next == null) return head;        
        ListNode nextNode = rotateHelp(head.next);
        while(K >= count) {
            //将多余的循环减去
            K -= count;
        }
        if(K > 0) {            
            nextNode.next = H;
            head.next = null;
            H = nextNode;
            K--;  
            if(K > 0) return head;         
        }
        return H;
    }
}
```

#### 解法二：空间换时间

用一个List按顺序存储节点。

```java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null || k == 0) return head;
        List<ListNode> list = new ArrayList<>();
        while(head != null) {
            list.add(head);
            head = head.next;
        }
        int count = list.size();
        while(k >= count) k -= count;
        if(k == 0) return list.get(0);
        list.get(count - 1).next = list.get(0);
        list.get(count - k - 1).next = null;
        return list.get(count - k);
    }
}
```

#### 解法三：快慢指针

经过上面的代码，发现并不需要存储那么多节点，关键的就是要保留尾结点、原头节点、新的头节点。

先用一个快指针移动k步，如果遇到k大于总节点数的情况，要对k作修改。

再快慢指针一起向前移动，直到快指针到了尾节点。将快指针和原头节点相连，再断开慢指针和慢指针下一个节点（新头节点）的连接，再返回新的头节点。

```java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null || k == 0) return head;
        ListNode slow = head, fast = head;
        int count = 1;
        for(int i = 0; i < k; i++) {
            if(fast.next != null){
                fast = fast.next;
                count++;
            }else {
                fast = head;
                while(k >= count) {
                    k -= count;
                    i = -1;
                }
            }
        }
        if(slow == fast) return slow;
        while(fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        fast.next = head;
        ListNode res = slow.next;
        slow.next = null;
        return res;
    }
}
```

### 62. 不同路径

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

**示例 1：**

![](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)

```
输入：m = 3, n = 7
输出：28
```


**示例 2：**

```
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。

1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
```

**示例 3：**

```
输入：m = 7, n = 3
输出：28
```

**示例 4：**

```
输入：m = 3, n = 3
输出：6
```

链接：https://leetcode-cn.com/problems/unique-paths

#### 解法一：深度优先搜索

每一个格子只有两种走法：向右或向下，就看这两个方向到达目标格子各有多少条路径。

```java
class Solution {
    public int uniquePaths(int m, int n) {
        return dfs(1,1,m,n);
    }
    private int dfs(int i, int j, int m, int n) {
        if(i > m || j > n) return 0;
        if(i == m && j == n) return 1;
        return dfs(i+1,j,m,n) + dfs(i,j+1,m,n);
    }
}
```

#### 解法二：动态规划

首先明确 dp[i] [j] 代表从起点到坐标为(i,j)的格子有多少种走法。这取决于其左边一个格子和上方一个格子的dp值，dp[i] [j] = dp[i-1] [j] + dp[i] [j-1]。 

初始化dp[0] [0] = 1

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        dp[0][0] = 1;
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(i == 0 && j == 0) continue;
                dp[i][j] = (i > 0 ? dp[i-1][j] : 0) + (j > 0 ? dp[i][j-1] : 0);
            }
        }
        return dp[m-1][n-1];
    }
}
```

上面用的是二维数组，但实际上这题还可以压缩成一维数组。

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[]dp = new int[n];
        dp[0] = 1;
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(i == 0 && j == 0) continue;
                dp[j] += (j > 0 ? dp[j-1] : 0);
            }
        }
        return dp[n-1];
    }
}
```

时间复杂度：O（m * n）

空间复杂度：O（n）

#### 解法三：公式

以上面的图为例，用R表示向右，D表示向下，写出路径看看规律：

RRRRRRDD、RRRRRDDR、RRRRDRDR、RRRRDDRR……

可以看出从左上角到右下角一定会向右走6次，向下走2次，只不过次序不一样，所以实质就是求一共m+n-2 = 8 步中挑6个向右走有多少种即$C^{m-1}_{m+n-2}$

因为$C^k_n = (n * (n-1) * (n-2) * ...*(n-k+1))/k!$可以写出下面的代码

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int N = m + n - 2;
        int k = m - 1;
        long res = 1;
        for(int i = 1; i <= k; i++) {
            res = res * (N - k + i) / i;
        }
        return (int)res;
    }
}
```

时间复杂度：O（m）

空间复杂度：O（1）

### 63. 不同路径II

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 1 和 0 来表示。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/11/04/robot1.jpg)

```
输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：

1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2020/11/04/robot2.jpg)

```
输入：obstacleGrid = [[0,1],[0,0]]
输出：1
```

链接：https://leetcode-cn.com/problems/unique-paths-ii

#### 解法一：深度优先搜索

和上一题一样的思路，只不过在遇到obstacleGrid[i] [j] = 1的时候也要返回0。

```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {        
        return dfs(obstacleGrid,0,0);
    }
    private int dfs(int[][] obstacleGrid, int i, int j) {
        if(i >= obstacleGrid.length || j >= obstacleGrid[0].length || obstacleGrid[i][j] == 1) return 0;
        if(i == obstacleGrid.length - 1 && j == obstacleGrid[0].length - 1) return 1;
        return dfs(obstacleGrid,i+1,j) + dfs(obstacleGrid,i,j+1);
    }
}
```

但是很不幸超出了时间限制。

#### 解法二：动态规划

和上面一题的动态规划思路也是一样的，只不过遇到了obstacleGrid[i] [j] = 1时，dp[i] [j] = 0。

```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {        
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        //如果障碍物在入口或出口则代表没有路径可以到达直接返回0
        if(obstacleGrid[0][0] == 1 || obstacleGrid[m-1][n-1] == 1) return 0;

        int[][] dp = new int[m][n];
        dp[0][0] = 1;//初始化入口值
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                //如果此处障碍物，则dp为0，又因为数组初始化直接赋值0所以可以不作任何操作。
                //而入口处赋过值也要跳过
                if(obstacleGrid[i][j] == 1 || (i==0 && j==0)) continue;
                dp[i][j] = (i>=1?dp[i-1][j]:0) + (j>=1?dp[i][j-1]:0);
            }
        }
        return dp[m-1][n-1];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（m*n）

这题也可以用一维数组来写。

```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {        
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        //如果障碍物在入口或出口则代表没有路径可以到达直接返回0
        if(obstacleGrid[0][0] == 1 || obstacleGrid[m-1][n-1] == 1) return 0;

        int[] dp = new int[n];
        dp[0] = 1;//初始化入口值
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                //入口处赋过值跳过
                if((i==0 && j==0)) continue;
                //如果此处障碍物，则dp为0，这里一定要复位0，不能向二维数组一样直接跳过。
                if(obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                }else{
                    dp[j] += j>=1 ? dp[j-1] : 0;
                }                
            }
        }
        return dp[n-1];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（n）

### 64. 最小路径和

给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

**示例 1：**

```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

**示例 2：**

```
输入：grid = [[1,2,3],[4,5,6]]
输出：12
```

链接：https://leetcode-cn.com/problems/minimum-path-sum

#### 解法一：动态规划

明确dp[i] [j]的含义：到达(i,j)的最小路径和。

又因为到达(i,j)只可能是从(i-1,j)或(i,j-1)过来的，那么dp[i] [j] = Math.min(dp[i-1] [j], dp[i] [j-1]) + grid[i] [j]。

可以直接利用grid数组作为dp

```java
class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(i == 0 && j == 0) continue;
                int prev = (Math.min(i >= 1 ? grid[i-1][j] : Integer.MAX_VALUE, j >= 1? grid[i][j-1] : Integer.MAX_VALUE));
                grid[i][j] += prev;
            }
        }
        return grid[m-1][n-1];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（1）

#### 解法二：递归

```java
class Solution {
    public int minPathSum(int[][] grid) {
        return dfs(grid,0,0);
    }
    private int dfs(int[][] grid, int i, int j) {
        if(i >= grid.length || j >= grid[0].length) return Integer.MAX_VALUE;
        if(i == grid.length - 1 && j == grid[0].length - 1) return grid[i][j];
        return grid[i][j] + Math.min(dfs(grid,i+1,j),dfs(grid,i,j+1));
    }
}
```

又 超出时间限制了。

### 69. x的平方根

实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

**示例 1:**

```
输入: 4
输出: 2
```

**示例 2:**

```
输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。
```

链接：https://leetcode-cn.com/problems/sqrtx

#### 解法一：二分法

求 n 的平方根的整数部分，所以平方根一定是 1，2，3 ... n 中的一个数。从一个有序序列中找一个数，可以直接用二分查找。先取中点 mid，然后判断 mid * mid 是否等于 n，小于 n 的话取左半部分，大于 n 的话取右半部分，等于 n 的话 mid 就是我们要找的。如果跳出了循环，那么最终的r就是我们要找的。

这里因为mid * mid有超过int范围的危险，所以改用n/mid与mid 的大小比较。

```java
class Solution {
    public int mySqrt(int x) {
        int mid = 0, l = 1, r = x;
        while(l <= r) {
            mid = l + ((r - l)/2);
            int res = x / mid; 
            if(res == mid) {
                return mid;
            }else if (res < mid){
                r = mid - 1;
            }else {
                l = mid + 1;
            }
        }
        return r;
    }
}
```

时间复杂度：O（log n）

空间复杂度：O（1）

#### 解法二：牛顿迭代法

牛顿迭代法详解看[这里](https://blog.csdn.net/ccnt_2012/article/details/81837154)。

这题相当于求$f(a) = a^2 - x$的值，根据牛顿迭代式$x_{n+1} = x_n - f(x_n)/f^{'}(x_n)$ 可以写出下面代码

```java
class Solution {
    public int mySqrt(int x) {
        //牛顿迭代法
        long a = x;
        while (a * a > x) {
            a = (a + x / a) / 2;
        }
        return ((int) a);
    }
}
```

### 70. 爬楼梯

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

注意：给定 n 是一个正整数。

**示例 1：**

```
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。

1.  1 阶 + 1 阶
2.  2 阶
```

**示例 2：**

```
输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。

1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶
```

链接：https://leetcode-cn.com/problems/climbing-stairs

#### 解法一：动态规划

很简单的动态规划

```java
class Solution {
    public int climbStairs(int n) {
        if(n <= 2) return n;
        int[] dp = new int[n+1];
        dp[0] = 1;
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= 2; j++) {
                if(i >= j) dp[i] += dp[i-j];
            }
        }
        return dp[n];
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

#### 解法二：矩阵快速幂

[官方题解的解法二](https://leetcode-cn.com/problems/climbing-stairs/solution/pa-lou-ti-by-leetcode-solution/)

对于状态$f(n+1)$取决于$f(n)+f(n-1)$，可以构建一个递推关系：
$$
\begin{bmatrix}1 & 1 \\ 1 & 0\\ \end{bmatrix} \begin{bmatrix}f(n) \\ f(n-1)\\ \end{bmatrix} = \begin{bmatrix}f(n) + f(n-1) \\ f(n)\\ \end{bmatrix} = \begin{bmatrix}f(n+1) \\ f(n)\\ \end{bmatrix}
$$

$$
\begin{bmatrix}f(n+1) \\ f(n)\\ \end{bmatrix} = \begin{bmatrix}1 & 1 \\ 1 & 0\\ \end{bmatrix}^n \begin{bmatrix}f(1) \\ f(0)\\ \end{bmatrix}
$$

令：
$$
M = \begin{bmatrix}1 & 1 \\ 1 & 0\\ \end{bmatrix}
$$
计算M的n次幂后就可以得到f(n)。求M的n次幂可以参考50题的解法二。

```java
public class Solution {
    public int climbStairs(int n) {
        int[][] q = {{1, 1}, {1, 0}};
        int[][] res = pow(q, n);
        return res[0][0];
    }

    public int[][] pow(int[][] a, int n) {
        int[][] ret = {{1, 0}, {0, 1}};
        while (n > 0) {
            if ((n & 1) == 1) {
                ret = multiply(ret, a);
            }
            n >>= 1;
            a = multiply(a, a);
        }
        return ret;
    }

    public int[][] multiply(int[][] a, int[][] b) {
        int[][] c = new int[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
            }
        }
        return c;
    }
}
```

时间复杂度：O（log n）

空间复杂度：O（1）

如果一个递归式形如$f(n) = \sum_{i=1}^{m}a_if(n-i)$，即齐次线性递推式，就可以把递推关系转化成矩阵的递推关系。矩阵如下：
$$
\begin{bmatrix}a_1 & a_2 & a_3 & ... &a_m \\ 1 & 0 & 0 & ... & 0\\0 & 1 & 0 & ... & 0\\0 & 0 & 1 & ... & 0\\. &. &.&.&.\\0 & 0&0 & ...&1 \end{bmatrix}
$$

#### 解法三：通项公式

因为$f(n+1)=f(n)+f(n-1)$，可以写出特征方程：$x^2 = x+1$，求出解$x_1={{1+\sqrt 5} \over 2}$和$x_1={{1-\sqrt 5} \over 2}$，设通解为$f(n)=c_1 x_1^2 + c_2x_2^2$，带入初始条件$f(1)=1,f(2) = 1$，求出$c_1=-{1 \over \sqrt 5},c_2={1 \over \sqrt 5}$，最终得到地推数列的通项公式为：
$$
f(n)={1 \over \sqrt 5}[({{1+\sqrt 5} \over 2})^n - ({{1-\sqrt 5} \over 2})^n]
$$

```java
public class Solution {
    public int climbStairs(int n) {
        double sqrt5 = Math.sqrt(5);
        double fibn = Math.pow((1 + sqrt5) / 2, n + 1) - Math.pow((1 - sqrt5) / 2, n + 1);
        return (int) Math.round(fibn / sqrt5);
    }
}
```

时间复杂度：与使用的pow函数的复杂度有关。

空间复杂度：O（1）

### 72. 编辑距离

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符

**示例 1：**

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2：**

```
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

链接：https://leetcode-cn.com/problems/edit-distance

#### 解法：动态规划

dp[i] [j]表示word1的前i个位置变到和word2的前j个位置都相等需要的最小操作数。

当`word1[i] == word2[j]`时，`dp[i][j] = dp[i-1][j-1]`

不相等时，有三种情况：

- 修改当前word1的i位置字母，`dp[i][j] = dp[i-1][j-1]+1`
- 删除当前i位置字母，`dp[i][j] = dp[i-1][j]+1`
- 增加字母，`dp[i][j] = dp[i][j-1]+1`

取这当中的最小值。

<img src="https://pic.leetcode-cn.com/76574ab7ff2877d63b80a2d4f8496fab3c441065552edc562f62d5809e75e97e-Snipaste_2019-05-29_15-28-02.png" style="zoom:80%;" />

针对第一行，第一列要单独考虑，引入 `''` ，表示空。

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length(), len2 = word2.length();
        if(len1 == 0) return len2;
        if(len2 == 0) return len1;
        //dp[i][j]表示word1的前i个位置变到和word2的前j个位置都相等需要的步数
        int[][] dp = new int[len1+1][len2+1];
        char[] Word1 = word1.toCharArray();
        char[] Word2 = word2.toCharArray();
        //初始化0行和0列
        for(int i = 0; i <= len1; i++) {
            dp[i][0] = i;
        }
        for(int i = 0; i <= len2; i++) {
            dp[0][i] = i;
        }
        for(int i = 1; i <= len1; i++) {
            for(int j = 1; j <= len2; j++) {
                if(Word1[i-1] == Word2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                }else {
                    //从dp[i-1][j-1]是代表替换当前字母
                    //从dp[i-1][j]是代表当前操作要删除字母
                    //从dp[i][j-1]是代表当前操作要增加字母
                    dp[i][j] = Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1])) + 1;
                }
            }
        }
        return dp[len1][len2];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（m*n）

空间可以再优化一下，可以只用两个数组，一个保存dp[i-1]，一个是dp[i]。

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length(), len2 = word2.length();
        if(len1 == 0) return len2;
        if(len2 == 0) return len1;

        int[][] dp = new int[2][len2+1];
        char[] Word1 = word1.toCharArray();
        char[] Word2 = word2.toCharArray();
        //初始化0行
        for(int i = 0; i <= len2; i++) {
            dp[0][i] = i;
        }
        for(int i = 1; i <= len1; i++) {
            dp[i%2][0] = i;//初始化0列
            for(int j = 1; j <= len2; j++) {
                if(Word1[i-1] == Word2[j-1]) {
                    dp[i%2][j] = dp[(i+1)%2][j-1];
                }else {
                    dp[i%2][j] = Math.min(dp[(i+1)%2][j-1], Math.min(dp[(i+1)%2][j], dp[i%2][j-1])) + 1;
                }
            }
        }
        return dp[len1%2][len2];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（n）

在这基础上其实还可以再优化一下空间，因为到我们计算到`dp[i][j]`的时候，需要的i-1状态只是`dp[i-1][j-1]`，并不需要dp[i-1]的所有状态，只不过我们在计算`dp[i][j-1]`的时候会覆盖掉`dp[i-1][j-1]`，用一个变量保存一下即可。

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length(), len2 = word2.length();
        if(len1 == 0) return len2;
        if(len2 == 0) return len1;

        int[] dp = new int[len2+1];
        char[] Word1 = word1.toCharArray();
        char[] Word2 = word2.toCharArray();
        
        for(int i = 0; i <= len2; i++) {
            dp[i] = i;
        }
        int prev = dp[0];//保存dp[i-1][j-1]
        for(int i = 1; i <= len1; i++) {
            prev = dp[0];
            dp[0] = i;//初始化0列
            for(int j = 1; j <= len2; j++) {
                int temp;
                if(Word1[i-1] == Word2[j-1]) {
                    temp = prev;
                }else {
                    temp = Math.min(prev, Math.min(dp[j], dp[j-1])) + 1;
                }
                prev = dp[j];
                dp[j] = temp;
            }
        }
        return dp[len2];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（n）

### 75. 颜色分类

给定一个包含红色、白色和蓝色，一共 n 个元素的数组，**原地**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

**示例 1：**

```
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
```

**示例 2：**

```
输入：nums = [2,0,1]
输出：[0,1,2]
```

**示例 3：**

```
输入：nums = [0]
输出：[0]
```

**示例 4：**

```
输入：nums = [1]
输出：[1]
```

链接：https://leetcode-cn.com/problems/sort-colors

#### 解法一：

先遍历一次数组，计算0、1、2出现的次数，再从头遍历数组直接按个数顺序修改即可。

```java
class Solution {
    public void sortColors(int[] nums) {
        int count0 = 0, count1 = 0, count2 = 0;
        for(int num : nums) {
            if(num == 0) {
                count0++;
            }else if(num == 1) {
                count1++;
            }else {
                count2++;
            }
        }
        for(int i = 0; i < nums.length; i++) {
            if(count0 > 0) {
                nums[i] = 0;
                count0--;
            }else if(count1 > 0) {
                nums[i] = 1;
                count1--;
            }else if(count2 > 0) {
                nums[i] = 2;
                count2--;
            }
        }
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法二：

上面需要遍历两次数组，那么考虑是否可以只遍历一次？

用i指针记录当前位置，index0是应该放0的位置索引，index2是应该放2的位置索引。

因为0一定是在数组最前面部分，2一定是在数组最后面的部分，所以最开始的`index0=0，index2=nums.length - 1`，i从前往后遍历，如果遇到等于0就拿当前位置和index0进行交换，如果遇到2就和index2交换。

```java
class Solution {
    public void sortColors(int[] nums) {
        int i = 0, index0 = 0, index2 = nums.length - 1;
        while(i <= index2 && index0 < index2) {
            if(nums[i] == 2) {
                //这里之所以不进行i++,是因为与后方还未判断的数字进行交换后，当前位置为何值仍需要判断，而如果是与index0交换，index0的值一定是已经判断过的数值                
                exch(nums,i,index2--);
            }else if(nums[i] == 0) {
                //这里i需要i++
                exch(nums,i++,index0++);
            }else {
                //这里遇到1不需要进行交换等操作，因为交换0和2已经能保证所有0一定在数组最前端那部分，2一定在数组最后端那一部分，那么自然1就会聚集在数组中间
                i++;
            }
        }
    }
    private void exch(int[] nums, int i, int j) {
        int store = nums[i];
        nums[i] = nums[j];
        nums[j] = store;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法三：

上面解法二虽然经典好用，但是如果遇到了五个数的情况就不适用了。

用三个指针分别表示三个数的末尾位置。

```
0  0  1  2  2  2  0  2  1
   ^  ^        ^  ^
  n0 n1       n2  i
```

然后当前遍历到 i 的位置，等于 0，我们只需要把 n2 指针后移并且将当前数字置为 2，将 n1 指针后移并且将当前数字置为 1，将 n0 指针后移并且将当前数字置为 0。

```
0  0  1  2  2  2  2  2  1  n2 后移后的情况 
   ^  ^           ^  
   n0 n1          i
                  n2  

0  0  1  1  2  2  2  2  1  n1 后移后的情况
   ^     ^        ^  
   n0    n1       i
                  n2                   

0  0  0  1  2  2  2  2  1  n0 后移后的情况
      ^  ^        ^  
      n0 n1       i
                  n2
```

```java
class Solution {
    public void sortColors(int[] nums) {
        int n0 = -1, n1 = -1, n2 = -1;      
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] == 0) {
                n2++;
                nums[n2] = 2;
                n1++;
                nums[n1] = 1;
                n0++;
                nums[n0] = 0;
            }else if(nums[i] == 1) {
                n2++;
                nums[n2] = 2;
                n1++;
                nums[n1] = 1;
            }else {
                n2++;
                nums[n2] = 2;
            }
        }
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 76. 最小覆盖子串

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。 

**示例 1：**

输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
**示例 2：**

输入：s = "a", t = "a"
输出："a"
链接：https://leetcode-cn.com/problems/minimum-window-substring

#### 解法：滑动窗口

不会写，看[题解](https://leetcode-cn.com/problems/minimum-window-substring/solution/tong-su-qie-xiang-xi-de-miao-shu-hua-dong-chuang-k/)。

用两个指针，一个指向子字符串的开头，一个指向结尾。搜寻过程中一旦发现目前的字符串已经包含了t中所有字母，就挪动开头指针，尾指针继续向后搜索。

思路非常巧妙的一道题。

```java
class Solution {
    public String minWindow(String s, String t) {
        int start = 0;
        int end = 0;
        int length = s.length();
        int count = t.length(); //总共还要寻找的字母数,这个变量可以帮助省去遍历ziMu数组
        int size = Integer.MAX_VALUE;//用于记录s中包含t所有字符的子字符串最小长度
        int[] ziMu = new int[128]; //每个字母还要再找到几个
        int pos = 0;//用于记录最终返回字符串的起始点

        for (int i = 0; i < count; i++){
            ziMu[t.charAt(i)]++;
        }

        while(end < length) {
            if(ziMu[s.charAt(end)] > 0) {
                count --;
            }
            ziMu[s.charAt(end)]--;

            if(count == 0) {
                while(end != start && ziMu[s.charAt(start)] < 0) {
                    /*解释ziMu数组：
                    当count == 0时，代表已经找到了区间的最小右端点，现在需要寻找最大左端点。即现区间中，多余的那些字母
                    在ziMu数组中对应的数量应当为负数，而最大左端点所在的位置的字符应当是t中包含的，且是首次遇到，那么对
                    应ziMu数组中的数量应该恰好为0。所以start要一直往左移动到恰好数组索引到的值为0。
                    */
                    ziMu[s.charAt(start++)]++;
                }

                if (end - start + 1 < size) {
                    size = end - start + 1;//记录下当前寻找到的最小区间字符数量
                    pos = start;
                }
				//左指针往左移动，即将开始新一轮搜寻，相应的这个字母应该在下一次区间里再找到一次
                ziMu[s.charAt(start++)]++;
                count++;//应当寻找的字母数也要增加
            }
            end++;
        }
        return size == Integer.MAX_VALUE?"":s.substring(pos,pos + size);
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 77. 组合

给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。

**示例:**

```
输入: n = 4, k = 2
输出:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

链接：https://leetcode-cn.com/problems/combinations

#### 解法一：回溯

一看到组合就知道是非常经典的回溯法了，需要注意的一点是这里还用到了剪枝，非常 巧妙。

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> combine(int n, int k) {
        List<Integer> path = new ArrayList<Integer>();
        backtracking(n, 1, k, path);
        return result;
    }
    private void backtracking(int n, int start, int k, List<Integer> path) {
        if(k == 0) {
            result.add(new ArrayList<Integer>(path));
            return;
        }
        //剪枝关键在于n-k+1，以样例[1,2,3,4],target=2为例，在第一层并不需要对4进行搜索
        for(int i = start; i <= n - k + 1; i++) {
            path.add(i);
            backtracking(n, i + 1, k - 1, path);
            path.remove(path.size() - 1);
        }
    }
}
```

#### 解法二：迭代

看官方题解许久，才搞明白他在干什么，又是二进制又是啥一大堆的...讲的过于啰嗦了。

思路其实不难，以n=4,k=2为例，最开始的组合是（1,2），比2小的数字只有1，所以以2为最高位的所有组合已经列举完，就给2+1，得到3，接下来要求以3为最高位的排列，有（1,3）和（2,3），2是比3小的最大的数字，所以3又要加一，求以4为最高位的排列（1,4）（2,4）（3,4）。

再以n=4,k=3为例，最开始求以3为最高位的组合（1,2,3），然后求以4为最高位的组合，在这个前提下，先求以2为第二高位的组合（1,2,4），求以3为第二高位的组合（1,3,4）（2,3,4）。

那么怎么样知道以数x为最高位的所有组合已经求完了呢？就是比它低一位的数仅比它小1，那么接下来就要求以x+1为最高位的组合，在低一位的数没有达到x-1之前，要对它进行加一。

设置一个长度为k+1的数组，前k位存放（1,2,...,k），第k+1位存放n+1。从低位往高位扫描，如果发现某一位的数字加一不等于其更高一位的数字，那么就要对它加一。

```java
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> path = new ArrayList<Integer>();
        for(int i = 1; i <= k; i++) {
            path.add(i);
        }
        path.add(n+1);
        int j = 0;
        while(j < k) {
            result.add(new ArrayList<Integer>(path.subList(0,k)));
            j = 0;
            
            while(j < k && path.get(j)+1 == path.get(j+1)) {
                path.set(j,j+1);
                j++;
            }
            
            path.set(j,path.get(j)+1);
        }
        return result;
    }
}
```

#### 解法三：迭代

以n=5,k=3为例，看图

![](https://windliang.oss-cn-beijing.aliyuncs.com/77_2.jpg)

第 1 次循环，我们找出所有 1 个数的可能 [ 1 ]，[ 2 ]，[ 3 ]。4 和 5 不可能，解法一分析过了，因为总共需要 3 个数，4，5 全加上才 2 个数。

第 2 次循环，在每个 list 添加 1 个数， [ 1 ] 扩展为 [ 1 , 2 ]，[ 1 , 3 ]，[ 1 , 4 ]。[ 1 , 5 ] 不可能，因为 5 后边没有数字了。 [ 2 ] 扩展为 [ 2 , 3 ]，[ 2 , 4 ]。[ 3 ] 扩展为 [ 3 , 4 ]；

第 3 次循环，在每个 list 添加 **1 个数（比list末尾数字大的所有数）**， [ 1，2 ] 扩展为[ 1，2，3]， [ 1，2，4]， [ 1，2，5]；[ 1，3 ] 扩展为 [ 1，3，4]， [ 1，3，5]；[ 1，4 ] 扩展为 [ 1，4，5]；[ 2，3 ] 扩展为 [ 2，3，4]， [ 2，3，5]；[ 2，4 ] 扩展为 [ 2，4，5]；[ 3，4 ] 扩展为 [ 3，4，5]；

最后结果就是，[[ 1，2，3]， [ 1，2，4]， [ 1，2，5]，[ 1，3，4]， [ 1，3，5]， [ 1，4，5]， [ 2，3，4]， [ 2，3，5]，[ 2，4，5]， [ 3，4，5]]。

```java
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        //添加第一个数
        for(int i = 1; i <= n - k + 1; i++) {
            result.add(Arrays.asList(i));
        }
        //添加其余的数
        for(int i = 2; i <= k; i++) {
            List<List<Integer>> tmp = new ArrayList<List<Integer>>();
            //遍历每个列表
            for(List<Integer> list : result) {
                for(int m = list.get(list.size() - 1) + 1; m <= n - (k - i); m++) {
                    List<Integer> newList = new ArrayList<Integer>(list);
                    newList.add(m);
                    tmp.add(newList);
                }
            }
            result = tmp;
        }
        return result;
    }
}
```

#### 解法四：递归

在n个里面选k个可以分解为，选定n之后在n-1里面选k-1，或是在n-1里面选k个。

```java
class Solution {
    public List<List<Integer>> combine(int n, int k) {       
        if(n == k || k == 0) {
            List<Integer> path = new ArrayList<Integer>();
            for(int i = 1; i <= k; i++) {
                path.add(i);
            }
            return new ArrayList<>(Arrays.asList(path));
        }

        //n-1里面选k-1个
        List<List<Integer>> result = combine(n - 1,k - 1);
        //每个结果加上 n
        result.forEach(e -> e.add(n));
        //n-1个里面选k个的结果也加入
        result.addAll(combine(n - 1, k));
        return result;
    }
}
```

这就是一种分治的思想，我平常很少能想到。

### 78. 子集

给你一个整数数组 nums ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**示例 2：**

```
输入：nums = [0]
输出：[[],[0]]
```

链接：https://leetcode-cn.com/problems/subsets

#### 解法一：回溯法

相当于分别求长度为0、1、2、...的组合，利用前面回溯法组合的代码即可完成。

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    int length;
    public List<List<Integer>> subsets(int[] nums) {
        length = nums.length;
        for(int i = 0; i <= length; i++) {
            backTracking(nums, 0, i);
        }
        return result;
    }
    private void backTracking(int[] nums, int start, int howmany) {
        if(howmany == 0){
            result.add(new ArrayList<>(path));
            return;
        }
        for(int i = start; i < length - howmany + 1; i++) {
            path.add(nums[i]);
            backTracking(nums, i + 1, howmany - 1);
            path.remove(path.size() - 1);
        }
    }
}
```

#### 解法二：回溯法

直接每次修改path都加入总结果result。

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
 
    public List<List<Integer>> subsets(int[] nums) {
        backTracking(nums, 0);
        return result;
    }
    
    private void backTracking(int[] nums, int start) {
        result.add(new ArrayList<>(path));
        for(int i = start; i < nums.length; i++) {
            path.add(nums[i]);
            backTracking(nums, i + 1);
            path.remove(path.size() - 1);
        }
    }
}
```

#### 解法三：迭代法

和77题解法三一样的思路。

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        result.add(new ArrayList<>());
        for(int i = 0; i < nums.length; i++) {
            result.add(Arrays.asList(nums[i]));
        }
        for(int i = 1; i < nums.length; i++) {
            int size = result.size();
            for(int m = 0; m < size; m++) {
                if(result.get(m).size() != i) continue;
                List<Integer> list = result.get(m);
                for(int j = 0; j < nums.length; j++) {
                    if(list.get(list.size() - 1) >= nums[j]) continue;
                    List<Integer> newList = new ArrayList<Integer>(list);
                    newList.add(nums[j]);
                    result.add(newList);
                }
            }
        }
        return result;
    }
}
```

#### 解法四：迭代二进制

这个思想很妙。

数组的每个元素，可以有两个状态，**在**（用1表示）子数组中和**不在**（用0表示）子数组中，所有状态的组合就是所有子数组了。

例如，nums = [ 1, 2 , 3 ]。1 代表在，0 代表不在。

```
3 2 1 			 代表的数字
0 0 0 -> [     ]	0
0 0 1 -> [    1]	1	
0 1 0 -> [  2  ]    2
0 1 1 -> [  2 1]  	3
1 0 0 -> [3    ]	4
1 0 1 -> [3   1] 	5
1 1 0 -> [3 2  ]	6
1 1 1 -> [3 2 1]	7
```

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        int n = nums.length;
        int max_bit = 1 << n; //总共有多少种组合
        for(int i = 0; i < max_bit; i++) {
            List<Integer> temp = new ArrayList<>();
            int count = 0;
            int i_copy = i; //不能直接用i！！！否则整个for循环会一直死循环的
            while(i_copy > 0) {
                if((i_copy & 1) == 1) {//如果当前最低位是1
                    temp.add(nums[count]);
                }
                i_copy >>= 1; //右移一位
                count++;//位++
            }
            result.add(temp);
        }
        return result;
    }
}
```

### 79. 单词搜索

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
**示例 2：**

![](https://assets.leetcode.com/uploads/2020/11/04/word-1.jpg)

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true
**示例 3：**

![](https://assets.leetcode.com/uploads/2020/10/15/word3.jpg)

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
链接：https://leetcode-cn.com/problems/word-search

#### 解法：深度优先搜索+回溯

看到这题很自然会想到深度优先搜索，但是由于题目要求了不能重复利用格子，所以要用一个布尔数组来记录该格子有没有被用过，回退的时候，这个字母就取消了被利用的状态所以布尔数组也要修改状态，这就是回溯的思想。

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        //如果word的长度大于board所有格子的数量那么一定无正确结果
        if(word.length() > board.length*board[0].length) return false; 
        //visited数组用来记录board中的某个字母是否已经被纳入了此轮搜索。
        boolean[][] visited = new boolean[board.length][board[0].length];
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                if(backTracing(board,word,visited,i,j,0)) return true;
            }
        }
        return false;
    }
	private boolean backTracing(char[][] board, String word, boolean[][] visited, int i, int j, int k) {
		if(i < 0 || i >= board.length || j < 0 || j >= board[0].length || visited[i][j] || board[i][j] != word.charAt(k)) {
			return false;
		}
        if (k == word.length() - 1) {
            //达到word末尾
			return true;
		}
        visited[i][j] = true;//当前字母已经被纳入了此轮搜索
        //四个方向上搜索
        boolean get = backTracing(board,word,visited,i-1,j,k+1) || backTracing(board,word,visited,i+1,j,k+1) || backTracing(board,word,visited,i,j-1,k+1) || backTracing(board,word,visited,i,j+1,k+1);
        //如果找到了直接返回true
        if(get) return true;
        //没找到的话要回退状态，把visited数组改回false
        visited[i][j] = false;
        return false;
	}
}
```

提交代码后发现速度很慢，执行时间是92 ms，看了别人的代码发现可以做一些剪枝操作，因为可能board中字母的数量根本就不满足构成word。所以先对他们分别进行频率统计。

剪枝后代码明显快了很多，执行时间减少到1 ms。

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        //如果word的长度大于board所有格子的数量那么一定无正确结果
        if(word.length() > board.length*board[0].length) return false; 
        //剪枝 频率统计
        int[] freq1 = new int[255];
        int[] freq2 = new int[255];
        for (int i = 0; i < word.length(); i++) {
            freq1[word.charAt(i)]++;
        }
        for (char[] chars : board) {
            for (char aChar : chars) {
                freq2[aChar]++;
            }
        }
        for (int i = 0; i < freq1.length; i++) {
            if (freq1[i]>freq2[i]){
                return false;
            }
        }
        //visited数组用来记录board中的某个字母是否已经被纳入了此轮搜索。
        boolean[][] visited = new boolean[board.length][board[0].length];
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                if(backTracing(board,word,visited,i,j,0)) return true;
            }
        }
        return false;
    }
	private boolean backTracing(char[][] board, String word, boolean[][] visited, int i, int j, int k) {
		if(i < 0 || i >= board.length || j < 0 || j >= board[0].length || visited[i][j] || board[i][j] != word.charAt(k)) {
			return false;
		}
        if (k == word.length() - 1) {
            //达到word末尾
			return true;
		}
        visited[i][j] = true;//当前字母已经被纳入了此轮搜索
        //四个方向上搜索
        boolean get = backTracing(board,word,visited,i-1,j,k+1) || backTracing(board,word,visited,i+1,j,k+1) || backTracing(board,word,visited,i,j-1,k+1) || backTracing(board,word,visited,i,j+1,k+1);
        //如果找到了直接返回true
        if(get) return true;
        //没找到的话要回退状态，把visited数组改回false
        visited[i][j] = false;
        return false;
	}
}
```

### 80. 删除有序数组中的重复项 II

给你一个有序数组 nums ，请你 **原地** 删除重复出现的元素，使每个元素 **最多出现两次** ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 **原地 修改输入数组** 并在使用 O(1) 额外空间的条件下完成。

说明：

为什么返回数值是整数，但输出的答案是数组呢？

请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:

```
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

**示例 1：**

```
输入：nums = [1,1,1,2,2,3]
输出：5, nums = [1,1,2,2,3]
解释：函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3 。 不需要考虑数组中超出新长度后面的元素。
```

**示例 2：**

```
输入：nums = [0,0,1,1,1,1,2,3,3]
输出：7, nums = [0,0,1,1,2,3,3]
解释：函数应返回新长度 length = 7, 并且原数组的前五个元素被修改为 0, 0, 1, 1, 2, 3, 3 。 不需要考虑数组中超出新长度后面的元素。
```

链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii

#### 解法一：快慢指针

两个指针，快指针一直向前移动，只有当fast所指向的数字遇到的次数count不大于2次的时候，slow才向前移动。

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int slow = 0;
        int count = 0;
        for(int fast = 0; fast < nums.length; fast++) {
            //计算fast指针指向的数字遇到的次数
            if(fast > 0 && nums[fast] == nums[fast-1]) {
                count++;    
            }else {
                count = 1;
            }
            //如果次数小于等于2，就修改slow指针指向的数字，并移动slow指针
            if(count <= 2) nums[slow++] = nums[fast];
        }
        return slow;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法二：快慢指针

上面的写法计算了count，但是实际上并不需要计算count，将当前fast指向的数字和slow指向的前2位比较，如果相等，因为有序，slow-1和slow-2都等于fast，再添加让slow=fast的话就超过 2 个了，所以不添加，如果不相等，那么就添加。

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        if(nums.length <= 2) return nums.length;
        int slow = 2;
        for(int fast = 2; fast < nums.length; fast++) {
            if(nums[fast] != nums[slow-2]) {
                nums[slow++] = nums[fast];
            }
        }
        return slow;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 81. 搜索旋转排序数组 II

已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 **旋转** ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 **从 0 开始** 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。

给你 **旋转后** 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。

**示例 1：**

```
输入：nums = [2,5,6,0,0,1,2], target = 0
输出：true
```

**示例 2：**

```
输入：nums = [2,5,6,0,0,1,2], target = 3
输出：false
```

链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii

#### 解法：二分法

比起33题，更难一点的地方爱与，数组中的值可能重复，像遇到了 nums = [ 1, 3, 1, 1, 1 ] ，target = 3，如果按照33题的写法会返回 false。因为在判断哪段有序的时候，当 nums [ start ] <= nums [ mid ] 是认为左半段有序。而由于这道题出现了重复数字，此时的 nums [ start ] = 1, nums [ mid ] = 1，但此时左半段 [ 1, 3, 1 ] 并不是有序的，所以造成算法错误。

所以当nums[mid]和两端值相等的时候要单独判断，只需要让两端的指针往中间移动一步即可。

```java
class Solution {
    public boolean search(int[] nums, int target) {
        int l =0,r=nums.length-1;
        while(l<=r){
            int mid = l+(r-l)/2;
            if(nums[mid]==target)return true;
            else if(nums[mid]<nums[r]){//右边有序
                if(nums[mid]<target && target<=nums[r])l=mid+1;
                else r=mid-1;
            }
            else if(nums[mid]>nums[r]){//左边有序
                if(nums[mid]>target && target>=nums[l])r=mid-1;
                else l=mid+1;
            }
            else r--; //nums[mid]=nums[r]
        }
        return false;
    }
}
```

时间复杂度：最好的情况，如果没有遇到 nums [ start ] == nums [ mid ]，是 O（log（n））。如果是类似于这种 [ 1, 1, 1, 1, 1, 1 ] ，就是 O ( n ) 。

空间复杂度：O ( 1 )。

### 88. 合并两个有序数组

给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。

初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。

**示例 1：**

```
输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
```

**示例 2：**

```
输入：nums1 = [1], m = 1, nums2 = [], n = 0
输出：[1]
```

链接：https://leetcode-cn.com/problems/merge-sorted-array

#### 解法一：

最开始的写法是开了一个新的数组，长度为m+n，然后将两个数组合并到这个新数组中，最后新数组再被拷贝到nums1中。

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = 0;
        int p2 = 0;
        int[] nums = new int[m+n];
        if(m==0) {
            nums = nums2;
        }else if(n == 0) {
            nums = nums1;
        }else{
            while(p1 + p2 < m + n) {
                if(p1 < m && p2 < n) {
                    if(nums1[p1] < nums2[p2]) {
                        nums[p1 + p2] = nums1[p1];
                        p1++;
                    }else if(nums1[p1] > nums2[p2]){
                        nums[p1+p2] = nums2[p2];
                        p2++;
                    }else{
                        System.out.println(nums2[p2]);
                        System.out.println(nums2[p2]);
                        nums[p1+p2] = nums1[p1];
                        p1++;
                        nums[p1+p2] = nums2[p2];
                        p2++;
                    }
                }else if(p2 < n){
                    nums[p1+p2] = nums2[p2];
                    p2++;
                }else{
                    nums[p1+p2] = nums1[p1];
                    p1++;                    
                }
            }
        }
        System.arraycopy(nums,0,nums1,0,m+n);
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

另外开了个数组空间的做法本身就挺作弊的...

#### 解法二：

本来是想着用类似26题的做法，达到原地修改数组的效果，但是发现，从前往后修改的话必然会面临一个难题：即修改nums1[i]的话就会丢失掉这个值。

看了别人的做法，发现可以从后往前遍历修改，必然不会覆盖掉还没有检测过的nums1里的值，这样就不会丢失了。

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int pos = m-- + n-- - 1;
        while(m >= 0 && n >= 0) {
            nums1[pos--] = nums1[m] > nums2[n] ? nums1[m--] : nums2[n--];
        }
        //如果m>=0不需要做处理，在nums1里的值本身自然有序。
        while(n >= 0) {            
            nums1[pos--] = nums2[n--];
        }
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 90. 子集 II

给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。返回的解集中，子集可以按 **任意顺序** 排列。

**示例 1：**

```
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
```

**示例 2：**

```
输入：nums = [0]
输出：[[],[0]]
```

链接：https://leetcode-cn.com/problems/subsets-ii

#### 解法一：回溯法

和78不同的是，数组中会含有重复的元素。遇到重复元素问题，一般都是需要排序的，排好序后，光看当前数和前一个数是否相等是不行的，这样可能会忽略掉一些情况，以示例1为例，[1,2,2]和[2,2]就不会被加入。这里的处理方法和47题是一样的。

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    int length;
    boolean[] used;
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        length = nums.length;
        used = new boolean[length];
        for(int i = 0; i <= length; i++) {
            backTracking(nums, 0, i);
        }
        return result;
    }
    private void backTracking(int[] nums, int start, int howmany) {
        if(howmany == 0){
            result.add(new ArrayList<>(path));
            return;
        }
        for(int i = start; i < length - howmany + 1; i++) {
            if(i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;
            used[i] = true;
            path.add(nums[i]);
            backTracking(nums, i + 1, howmany - 1);
            path.remove(path.size() - 1);
            used[i] = false;
        }
    }
}
```

#### 解法二：迭代法

以上面的示例1为例，迭代法的顺序如下图

![](https://windliang.oss-cn-beijing.aliyuncs.com/90_2.jpg)

大循环以数组进行遍历，小循环遍历每一个列表，给每个列表加入当前这个数。但是因为数组内有重复的数字，有些列表是不需要添加的，如何判断呢？

观察上面发现，在第四轮的时候，遇到了当前的数2与上一次大循环遍历的数2相同的状况，对于列表[]和[1]是不需要再添加2的，因为在第三轮的时候已经添加过了，而第三轮添加的[2]和[1,2]列表需要再添加2。即第三轮添加过的列表在这一轮不需要在添加2，而第三轮添加2后得到的新列表需要再添加2。

那么用一个指针保存一下需要开始添加的列表的位置即可。

```java
class Solution {    
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        result.add(new ArrayList<Integer>());//加入空列表
        int start = 0;
        for(int i = 0; i < nums.length; i++) {//遍历nums数组
        //用tmp来存储要加入result中的列表，这样可以保证result的size暂时不发生变化
            List<List<Integer>> tmp = new ArrayList<>();
            for(int j = 0; j < result.size(); j++) {//遍历result列表
                if(i > 0 && nums[i] == nums[i-1] && j < start) continue;
                //新建列表，不可以直接在原来的列表里进行添加，这样会改变result里已有的列表
                List<Integer> list_tmp = new ArrayList<>(result.get(j));
                list_tmp.add(nums[i]);
                tmp.add(list_tmp);
            }
            start = result.size();
            result.addAll(tmp);
        }        
        return result;
    }
}
```

### 91. 解码方法

一条包含字母 A-Z 的消息通过以下映射进行了 **编码** ：

```
'A' -> 1
'B' -> 2
...
'Z' -> 26
```

要 **解码** 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，`"11106"` 可以映射为：

`"AAJF"` ，将消息分组为` (1 1 10 6)`
`"KJF"` ，将消息分组为` (11 10 6)`
注意，消息不能分组为 ` (1 11 06) `，因为 `"06"` 不能映射为` "F" `，这是由于` "6"` 和 `"06"` 在映射中并不等价。

给你一个只含数字的 非空 字符串` s` ，请计算并返回 **解码** 方法的 **总数** 。

题目数据保证答案肯定是一个 32 位 的整数。 

**示例 1：**

```
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
```

**示例 2：**

```
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
```

**示例 3：**

```
输入：s = "0"
输出：0
解释：没有字符映射到以 0 开头的数字。
含有 0 的有效映射是 'J' -> "10" 和 'T'-> "20" 。
由于没有字符，因此没有有效的方法对此进行解码，因为所有数字都需要映射。
```

**示例 4：**

```
输入：s = "06"
输出：0
解释："06" 不能映射到 "F" ，因为字符串含有前导 0（"6" 和 "06" 在映射中并不等价）。
```

链接：https://leetcode-cn.com/problems/decode-ways

#### 解法一：深度优先搜索

首先题目里一大堆对应关系都是废话，说白了就是构成的二位数字是合法地在1-26之间的数。

上来就想到了DFS，一顿猛操作写完，提交，好家伙，超时。

```java
class Solution {
    public int numDecodings(String s) {
        char[] chars = s.toCharArray();
        return dfs(chars, 0);
    }
    private int dfs(char[] chars, int start) {
        if(start == chars.length) return 1;
        if(chars[start] == '0') return 0;
        int num = 0;
        int count = 0;
        for(int i = start; i <= start + 1 && i < chars.length; i++) {
            num = num*10 + (int) (chars[i] - '0');
            System.out.println(num);
            if(num > 26) return count;
            count += dfs(chars, i+1);
        }
        return count;
    }
}
```

这是因为存在着重复计算，以“1111”为例，在计算第一个1的时候，递归到最后一位，已经把最后两个1能够构成有效编码的数量计算出来了，但是当第一个1和第二个1组成一个码的时候，又要重新计算后两个1能够构成的有效编码数量。

所以这题不能这么写，得用动态规划。

#### 解法二：动态规划

明确 `dp[i]` 的含义，是到i位置能够构成的编码总数。

i位置可以自己单独构成一个数字，也可以和前一个位置构成数字。但是要判断是否合法，例如单个数字要看是否为0，两位数字要看是否以0开头或者是否超出了26的范围。

```java
class Solution {
    public int numDecodings(String s) {
        char[] chars = s.toCharArray();
        if(chars[0] == '0') return 0;
        if(chars.length <= 1) return chars.length;
        int[] dp = new int[chars.length];
        for(int i = 0; i < chars.length; i++) {
            if(i == 0) {
                dp[0] = 1;
                continue;
            }
            
            if(chars[i] == '0') {
                if(chars[i-1] > '0' && chars[i-1] < '3') dp[i] = i > 1 ? dp[i-2] : 1;
                continue;
            }
            dp[i] = dp[i-1];
            if(chars[i-1] == '1' || (chars[i-1] == '2' && chars[i] < '7')) dp[i] += i > 1 ? dp[i-2] : 1;
        }

        return dp[chars.length-1];
    }    
}
```

时间复杂度：O（n）

空间复杂度：O（1）

只要知道这题是要用动态规划写，就不难。

### 92. 反转链表 II

给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。

**示例 1：**

![](https://assets.leetcode.com/uploads/2021/02/19/rev2ex2.jpg)

```
输入：head = [1,2,3,4,5], left = 2, right = 4
输出：[1,4,3,2,5]
```

**示例 2：**

```
输入：head = [5], left = 1, right = 1
输出：[5]
```

链接：https://leetcode-cn.com/problems/reverse-linked-list-ii

#### 解法一：快慢指针+递归

其实就是中间那一段反转链表，然后再把头尾部分和中间一段拼接到一起。所以可以先通过快慢指针截取到需要翻转的那一段，然后再利用反转整条链表的递归函数反转这一段，再连接到一起。

比较麻烦的是头节点的返回，可以通过设置一个伪头节点来解决。

```java
class Solution {
    ListNode store = null;//保存中间部分反转后的头节点
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if(left == right) return head;
        ListNode slow = head, fast = head; 
        ListNode fakeHead = new ListNode(0,head);
        int n = right - left;
        for(int i = 1; i <= n; i++) {
            fast = fast.next;
        }
        head = fakeHead;
        for(int i = 1; i < left; i++) {
            slow = slow.next;
            fast = fast.next;
            head = head.next;//移动head到slow的前一个节点
        }        
        ListNode p = fast.next;//指向fast后面的尾部链表
        fast.next = null;//把slow到fast之间的链表独立出来
        reverse(slow).next = p; //调用递归函数反转链表后得到反转链表的尾结点，与p拼接
        head.next = store;//把反转后的链表头节点和链表的头部分链表连接
        return fakeHead.next;//返回真正的头节点
    }
    private ListNode reverse(ListNode head) {
        if(head.next == null) {
            store = head;
            return head;
        }
        ListNode tmp = reverse(head.next);
        tmp.next = head;
        return head;
    }
}
```

#### 解法二：迭代

可以用一个数组来存储中间那一段链表的值，然后再迭代一遍，对每个节点进行值的修改，改成对称位置的节点值。

```java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if(left == right) return head;
        ListNode p1 = head, p2 = head;
        int n = right - left + 1;
        int[] nums = new int[n];
        for(int i = 1; i <= right; i++) {
            if(i >= left) {
                nums[i-left] = p1.val;
                if(i == left) p2 = p1; //存储中间段的头
            }
            p1 = p1.next;
        }
        for(int i = 0; i < n; i++) {
            p2.val = nums[n-1-i];
            p2 = p2.next;
        }
        return head;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（right - left）

#### 解法三：迭代

上面的写法其实都不满足题目进阶要求只遍历一遍。那么如何通过只遍历一遍就完成链表的反转？以题目的示例一做演示：

[1 2 3 4 5] 先找到开始反转部分的前一个节点，即1。并找到开始反转的节点2。

往后遍历到节点3，把3移到最前面使链表变成[1 3 2 4 5]，这需要先将3.next保留下来，然后将2.next连接上3.next，将1.next连上3，再将3.next连上2。

遍历到节点4，要把4移到最前面使链表变成[1 4 3 2 5]，要把原4.next保留下来，将2.next连上4.next，将1.next连上4，再将4.next连上3。

```java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if(left == right) return head;
        ListNode fakeHead = new ListNode(0,head);
        ListNode p1 = fakeHead;
        for(int i = 1; i < left; i++) {
            p1 = p1.next; //找到反转部分的前一个节点
        }
        head = p1.next;//head为反转部分的头
        ListNode p2 = head.next;
        for(int i = left; i < right; i++) {
            head.next = p2.next;
            ListNode tmp = p1.next;
            p1.next = p2;
            p2.next = tmp;
            p2 = head.next;
        }
        return fakeHead.next;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

反转部分的代码有点绕，画个图会好很多。

### 93. 复原IP地址

给定一个只包含数字的字符串，用以表示一个 IP 地址，返回所有可能从 s 获得的 有效 IP 地址 。你可以按任何顺序返回答案。

有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。

例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。

**示例 1：**

```
输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]
```

**示例 2：**

```
输入：s = "0000"
输出：["0.0.0.0"]
```

**示例 3：**

```
输入：s = "1111"
输出：["1.1.1.1"]
```

**示例 4：**

```
输入：s = "010010"
输出：["0.10.0.10","0.100.1.0"]
```

**示例 5：**

```
输入：s = "101023"
输出：["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
```

链接：https://leetcode-cn.com/problems/restore-ip-addresses

#### 解法：回溯法

看到题目后，很自然就想到了回溯法，一位一位地增加构成当前段。

```java
class Solution {
    List<String> result = new ArrayList<>();
    StringBuilder path = new StringBuilder();
    public List<String> restoreIpAddresses(String s) {
        char[] chars = s.toCharArray();
        backTracing(chars, 0, 0);
        return result;
    }
    private void backTracing(char[] chars, int start, int seg) {
        if(start == chars.length) {
            if(seg == 4) result.add(path.toString());
            return;
        }
        if(seg > 0) path.append(".");
        int number = 0;
        StringBuilder tmp = new StringBuilder();
        for(int i = start; i < start + 3 && i < chars.length; i++) {
            //计算当前段的数字
            number = number * 10 + (int) (chars[i]-'0');
            tmp.append(chars[i]);
            //如果当前的位数过少，不满足后面生成其他段，将不进行递归
            if(number > 0 && i < chars.length - (3-seg)*3 - 1) continue;
            //如果数字小于等于255，可以进行下一段的构成
            if(number <= 255) {
                //将目前组成的数字段挂到path后面
                path.append(tmp);
                backTracing(chars, i+1, seg+1);
                //回退的时候，要删除tmp，起始位置是start+seg，加上seg是因为有'.'
                path.delete(start+seg, path.length());
            }else {
                break;
            }
            //如果当前数字是0，那么将不继续与后面的数字构成当前段。
            if(number == 0) break;
        }
    }
}
```

下面是另一种回溯写法，用一个int数组来存储每一段的IP。

```java
class Solution {
    List<String> result = new ArrayList<>();//最后返回的结果
    int[] segments = new int[4];//存储每一段IP
    public List<String> restoreIpAddresses(String s) {
        if(s.length() < 4 || s.length() > 12) return result;
        backTacking(s,0,0);
        return result;
    }
    private void backTacking(String s, int depth, int start) {
        if(start == s.length()) {//当搜寻完所有数字时，一定会return，不论是否满足要求
            if(depth == 4) {//在搜寻完所有数字且已经搜寻完成四段时（即满足要求），要把他们存入结果中
                StringBuilder path = new StringBuilder();
                for(int m = 0; m < 3; m++) {
                    path.append(segments[m]);
                    path.append(".");//加分割点
                }
                path.append(segments[3]);//对最后一段不需要分割点的单独处理
                result.add(path.toString());
            }
            return;
        }
        if(depth == 4) return;//代表当已经满四段，且未达到末尾，也要出栈

        if(s.charAt(start) == '0') {
            //当搜寻开始位置为0，那么将不进行下面的for循环，直接作为单独一段
            segments[depth]=0;
            backTacking(s, depth+1, start+1);
        }else{
            for(int i = start; i < start+4; i++) {
                if(s.length() - i - 1 > 3*(3 - depth)) continue;//当剩下的字符数过多时要继续取字符
                if(s.length() - i - 1 < 3 - depth) break;//当剩下的字符数过少时不可以再进行for循环
                int addr = Integer.parseInt(s.substring(start, i+1));//取出一段字符转为int
                if(addr <= 255) {//比较与255的大小，如果比255大，那么将不可以再进行for循环
                    segments[depth] = addr;
                    backTacking(s, depth+1, i+1);//进行下一段的搜寻
                }else {
                    break;
                }
            }
        }
    }
}
```

### 94. 二叉树的中序遍历

给定一个二叉树的根节点 root ，返回它的 中序 遍历。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```
输入：root = [1,null,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：root = []
输出：[]
```

**示例 3：**

```
输入：root = [1]
输出：[1]
```

**示例 4：**

![](https://assets.leetcode.com/uploads/2020/09/15/inorder_5.jpg)

```
输入：root = [1,2]
输出：[2,1]
```

**示例 5：**

![](https://assets.leetcode.com/uploads/2020/09/15/inorder_4.jpg)

```
输入：root = [1,null,2]
输出：[1,2]
```

链接：https://leetcode-cn.com/problems/binary-tree-inorder-traversal

#### 解法一：递归

递归的写法很好理解，就是把加入结果这一句代码放在中间。

```java
class Solution {
    List<Integer> result = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        dfs(root);
        return result;
    }
    private void dfs(TreeNode nowNode) {
        if(nowNode!=null){
            dfs(nowNode.left);
            result.add(nowNode.val);
            dfs(nowNode.right);
        }
    }
}
```

时间复杂度：O（n），遍历每个节点。

空间复杂度：O（h），压栈消耗，h 是二叉树的高度。

#### 解法二：迭代

迭代是用栈来模拟递归调用的过程，递归调用的时候是一路往左走直到遇到空节点，返回，将当前节点加入结果列表，再向右一路走。

![](https://pic.leetcode-cn.com/47fff35dd3fd640ba60349c78b85242ae8f4b850f06a282cd7e92c91e6eff406-1.gif)

所以用栈的时候就可以先将左节点加入栈，直到左节点是空，再弹栈，并将右节点加入栈。

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if(root == null) {
            return result;
        }
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        TreeNode node = root;
        while(!stack.isEmpty() || node != null) {
            if(node != null) {
                stack.push(node);
                node = node.left;//一路往左
            }else {//遇到空节点
                node = stack.pop();//弹栈
                result.add(node.val);//加入链表
                node = node.right;//往右走
            }
        }
        return result;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（h）

#### 解法三：Morris Traversal

不用额外空间，强行把一棵二叉树改成一段链表结构。

把当前节点的右子树全部挂到左子树的最右边节点上。

![](https://pic.leetcode-cn.com/c1b589b5fc7facd1a847c9f5bab407765222ee2d9e1a887a9e5d61cc9e94dfc6-3.gif)

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if(root == null) {
            return result;
        }
        while(root != null) {
            if(root.left == null) {
                //左节点为空的时候，将当前节点加入结果队列
                result.add(root.val);
                //往右走
                root = root.right;
            }else {//左节点不为空的时候
                TreeNode pre = root.left;
                //找寻左子树的最右边节点
                while(pre.right != null) pre = pre.right;
                //将root连到该最右节点的右子树上
                pre.right = root;
                TreeNode tmp = root; 
                //将root设置为原root的左节点
                root = tmp.left;
                //断开原root和现root之间的连接
                tmp.left = null;
            }
        }
        return result;
    }
}
```

时间复杂度：O（n）。

空间复杂度：O（1）。

### 95. 不同的二叉搜索树 II

给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。 

**示例 1：**

![](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)

```
输入：n = 3
输出：[[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
```


**示例 2：**

```
输入：n = 1
输出：[[1]]
```

链接：https://leetcode-cn.com/problems/unique-binary-search-trees-ii

#### 解法一：分治

可以看做分别生成左子树和右子树。

求 1...n 的所有可能。

我们只需要把 1 作为根节点，[ ] 空作为左子树，[ 2 ... n ] 的所有可能作为右子树。

2 作为根节点，[ 1 ] 作为左子树，[ 3...n ] 的所有可能作为右子树。

3 作为根节点，[ 1 2 ] 的所有可能作为左子树，[ 4 ... n ] 的所有可能作为右子树，然后左子树和右子树两两组合。

```java
class Solution {
    public List<TreeNode> generateTrees(int n) {
        if (n < 1) {
            return new ArrayList<TreeNode>();
        }
        return generateTrees(1,n);
    }
    public List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> res = new ArrayList<>();
        if(start > end) res.add(null);
        for(int i = start; i <= end; i++) {
            List<TreeNode> left = generateTrees(start,i-1);
            List<TreeNode> right = generateTrees(i+1,end);            
            for(TreeNode l : left) {
                for(TreeNode r : right) {
                    TreeNode now = new TreeNode(i);                    
                    now.left = l;
                    now.right = r;
                    res.add(now);
                }
            }
        }
        return res;
    }
}
```

#### 解法二：动态规划

n = 0的时候，节点只能为null

n = 1，节点只有一个[1]。

n = 2，可以是以2为根节点[2,1]和以1为根节点[1,null,2]。

n = 3，可以作为根节点，连接n = 2的两种，得到[3, 2, null, 1] [3, 1, null, null, 2]，或者作为n = 2的两种情况的最右边节点，得到[2, 1, 3]和[1, null, 2, null, null, null, 3]，或者作为根节点的右孩子。

观察总结规律，每次新增加的数字大于之前的所有数字，所以新增加的数字出现的位置只可能是根节点或者是根节点的右孩子，右孩子的右孩子，右孩子的右孩子的右孩子等等，总之一定是右边。

```java
class Solution {
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> pre = new ArrayList<TreeNode>();
        if (n == 0) {
            return pre;
        }
        pre.add(null);
        //每次增加一个数字
        for (int i = 1; i <= n; i++) {
            List<TreeNode> cur = new ArrayList<TreeNode>();
            //遍历之前的所有解
            for (TreeNode root : pre) {
                //插入到根节点
                TreeNode insert = new TreeNode(i);
                insert.left = root;
                cur.add(insert);
                //插入到右孩子，右孩子的右孩子...最多找 n 次孩子
                for (int j = 0; j <= n; j++) {
                    TreeNode root_copy = treeCopy(root); //复制当前的树
                    TreeNode right = root_copy; //用于寻找要插入右孩子的位置
                    //遍历 j 次找右孩子
                    for (int k = 0; k < j; k++) {
                        if (right == null) break;
                        right = right.right;
                    }
                    //到达 null 提前结束
                    if (right == null) break;
                    //保存当前右孩子的位置的子树作为插入节点的左孩子
                    TreeNode rightTree = right.right;
                    insert = new TreeNode(i);
                    right.right = insert; //右孩子是插入的节点
                    insert.left = rightTree; //插入节点的左孩子更新为插入位置之前的子树
                    //加入结果中
                    cur.add(root_copy);
                }
            }
            pre = cur;
        }
        return pre;
    }
    private TreeNode treeCopy(TreeNode root) {
        if (root == null) {
            return root;
        }
        TreeNode newRoot = new TreeNode(root.val);
        newRoot.left = treeCopy(root.left);
        newRoot.right = treeCopy(root.right);
        return newRoot;
    }
}
```

这种写法是我没想到的，很值得学习。

### 96. 不同的二叉搜索树

给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

**示例 1：**

![](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)

```
输入：n = 3
输出：5
```

**示例 2：**

```
输入：n = 1
输出：1
```

链接：https://leetcode-cn.com/problems/unique-binary-search-trees

#### 解法一：动态规划

dp[i]含义是由1到i组成的二叉搜索树的可能性。

n = 0，只能是一个null

n = 1，只有一个节点[1]

n = 2，考虑左右各有多少种，要么是左1右0，要么是左0右1，共有两种情况。

n = 3，要么是左2右0，左1右1，左0右2，所以是2*1 + 1 * 1 + 1* 2 = 5。

得到当n = i，是左i-1右0，左i-2右1，左i-3右2，...左1右i-2，左0右i-1，$dp[i] = dp[i-1] * dp[0] + dp[i-2] * dp[1] + dp[i-3] * dp[2] + ... + dp[0] * dp[i-1]$。

```java
class Solution {
    public int numTrees(int n) {
        int dp[] = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= n; i++) {
            for(int j = 1; j <= i; j++) {
                dp[i] += dp[j-1]*dp[i-j];
            }
        }
        return dp[n];
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

#### 解法二：分治

和上一题的分治是一样的思想，但是超时了。

```java
class Solution {
    public int numTrees(int n) {
        return numTrees(1, n);
    }
    public int numTrees(int start, int end) {
        if(start >= end) return 1;
        int ans = 0;
        for(int i = start; i <= end; i++) {
            int left = numTrees(start, i-1);
            int right = numTrees(i+1, end);
            ans += left * right;
        }
        return ans;
    }
}
```

#### 解法三：公式法

卡塔兰树数列的定义：

> 令h ( 0 ) = 1，catalan 数满足递推式：
>
> **h ( n ) = h ( 0 ) \* h ( n - 1 ) + h ( 1 ) \* h ( n - 2 ) + ... + h ( n - 1 ) \* h ( 0 ) ( n >=1 )**
>
> 例如：h ( 2 ) = h ( 0 ) * h ( 1 ) + h ( 1 ) * h ( 0 ) = 1 * 1 + 1 * 1 = 2
>
> h ( 3 ) = h ( 0 ) * h ( 2 ) + h ( 1 ) * h ( 1 ) + h ( 2 ) * h ( 0 ) = 1 * 2 + 1 * 1 + 2 * 1 = 5

看上面动态规划的公式发现是契合的。卡塔兰数有一个通项公式：
$$
C_n = {{(2n)!}\over{(n+1)!n!}} =(2n)!/(n+1)!n!=(2n)∗(2n−1)∗...∗(n+1)/(n+1)!
$$
根据公式计算即可。

```java
class Solution {
    public int numTrees(int n) {
        long ans = 1;
        for(int i = 1; i <= n; i++) {
            ans = ans * (i + n) / i;
        }
        return (int)(ans/(n+1));
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 98. 验证二叉搜索树

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。
**示例 1:**

```
输入:
    2
   / \
  1   3
输出: true
```

**示例 2:**

```
输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。
```

链接：https://leetcode-cn.com/problems/validate-binary-search-tree

#### 解法一：中序遍历+全局变量

通过中序遍历将数值添加入列表，如果是合法的二叉搜索树应该是一个单调增的列表。

```java
class Solution {
    List<Integer> alist = new ArrayList<>();
    public boolean isValidBST(TreeNode root) {
        if(root == null) return true;
        mid(root);
        if(alist.size() == 1) return true;
        int prev = alist.get(0);
        for(int i = 1; i < alist.size(); i++) {
            int now = alist.get(i);
            if(prev >= now) return false;
            prev = now;
        }
        return true;
    }
    public void mid(TreeNode root) {
        if(root == null) {
            return;
        }
        mid(root.left);
        alist.add(root.val);
        mid(root.right);
    }
}
```

但是比较大小这件事实际上在递归的过程中就可以完成。

```java
class Solution {
    List<Integer> alist = new ArrayList<>();
    boolean flag= true;
    public boolean isValidBST(TreeNode root) {
        if(root == null) return true;
        mid(root);
        return flag;
    }
    public void mid(TreeNode root) {
        if(root == null) {
            return;
        }
        if(flag) {
            mid(root.left);
            int now = root.val;
            if(alist.size() > 0 && now <= alist.get(alist.size()-1)) {
                flag = false;
                return;
            }
            alist.add(now);
            mid(root.right);
        }
    }
}
```

观察发现，其实我们并不需要保存这么多数，只需要保存上一个数即可。

```java
class Solution {
    int prev = Integer.MIN_VALUE;
    boolean flag= true;
    public boolean isValidBST(TreeNode root) {
        if(root == null) return true;
        mid(root);
        return flag;
    }
    public void mid(TreeNode root) {
        if(root == null) {
            return;
        }
        if(flag) {
            mid(root.left);
            int now = root.val;
            if(prev >= now) {
                flag = false;
                return;
            }
            prev = now;
            mid(root.right);
        }
    }
}
```

这种写法的坏处在于不能够及时地停止递归，即使知道结果是false，还是得递归。

#### 解法二：前序遍历+局部变量

下面的写法就相当厉害了。清晰精炼。

要让一个树为合法的二叉搜索树，我们只要保证一个节点的值小于他的右父母，大于他的左父母，即我们要保证这三个值是单调递增的关系即可。通过两个形参，分别传递一个low一个up值，当前遍历的节点必须保证在这两个值之间。往左遍历节点的时候，必须要保证当前的节点值大于左孩子的值，所以递归的时候，就要将当前节点值作为up传给下面的函数。往右遍历的时候，反之。

```java
class Solution {
    public boolean isValidBST(TreeNode root) {        
        return isValidBST(root,Long.MIN_VALUE,Long.MAX_VALUE);
    }
    public boolean isValidBST(TreeNode node, long low, long up) {
        if(node == null) return true;
        if(node.val <= low || node.val >= up) return false;
        return isValidBST(node.left,low,node.val) && isValidBST(node.right,node.val,up);
    }
}
```

拆解一下，得到下面的中序写法可能更好理解

```java
class Solution {
    long pre = Long.MIN_VALUE;
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        // 访问左子树
        if (!isValidBST(root.left)) {
            return false;
        }
        // 访问当前节点：如果当前节点小于等于中序遍历的前一个节点，说明不满足BST，返回 false；否则继续遍历。
        if (root.val <= pre) {
            return false;
        }
        pre = root.val;
        // 访问右子树
        return isValidBST(root.right);
    }
}
```

观察后发现，其实本质就是上面解法一的改良。

这题首先还是要理清用哪种顺序，找到题目问的本质（单调增），然后就好写了。

### 100. 相同的树

给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

**示例 1：**

```
输入：p = [1,2,3], q = [1,2,3]
输出：true
```

**示例 2：**

```
输入：p = [1,2], q = [1,null,2]
输出：false
```

**示例 3：**

```
输入：p = [1,2,1], q = [1,1,2]
输出：false
```

链接：https://leetcode-cn.com/problems/same-tree

#### 解法一：前序遍历

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p == null || q == null) return false;
        if(p.val == q.val) return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);
        return false;
    }
}
```

#### 解法二：中序遍历

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p == null || q == null) return false;
        if(!isSameTree(p.left,q.left) || p.val != q.val) return false;
        return isSameTree(p.right,q.right);
    }
}
```

#### 解法三：后序遍历

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p == null || q == null) return false;
        if(!isSameTree(p.left,q.left) || !isSameTree(p.right,q.right)) return false;
        return p.val == q.val;
    }
}
```

### 101. 对称二叉树

给定一个二叉树，检查它是否是镜像对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

    	1
       / \
      2   2
     / \ / \
    3  4 4  3

但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

    	1
       / \
      2   2
       \   \
       3    3

链接：https://leetcode-cn.com/problems/symmetric-tree

#### 解法一：DFS

就是看根节点的左子树和右子树是否对称，下面的写法很容易想到

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root==null) return true;
        
        return dfs(root.left,root.right);
    }

    boolean dfs(TreeNode left,TreeNode right){
        if(left==null && right==null) return  true;
        
        if(left==null || right==null) return false;
        
        if(left.val != right.val) return false;
        
        return dfs(left.left,right.right)&&dfs(left.right,right.left);
    }
}
```

#### 解法二：DFS+链表

设置一个链表，存储每一层的所有数值，最终链表里的数会按层间遍历的顺序摆放，检查每一层对称位置的值是否相等即可。

```java
class Solution {
    List<List<Integer>> tree = new ArrayList<>();
    public boolean isSymmetric(TreeNode root) {
        dfs(root,0);
        int l1 = tree.size();
        for(int i = 0; i < l1; i++) {
            for(int j = 0; j < tree.get(i).size()/2; j++) {
                if(!tree.get(i).get(j).equals(tree.get(i).get(tree.get(i).size()-1-j))) return false;
            }
        }
        return true;
    }
    private void dfs(TreeNode node, int level) {        
        if(level >= tree.size()) {
            for(int i = tree.size(); i < level+1; i++) {
                tree.add(new ArrayList<>());
            }
        }
        if(node == null) {
            //遇到null给定一个值
            tree.get(level).add(Integer.MAX_VALUE-1);
            return;
        }else{
            //将当前节点值加入tree
            tree.get(level).add(node.val);
            //往左
            dfs(node.left,level+1);
            //往右
            dfs(node.right,level+1);
        }        
    }
}
```

#### 解法三：BFS+队列

按照层间遍历的顺序来遍历左右两边的树。

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root==null){
            return true;
        }
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root.left);
        que.offer(root.right);
        while(!que.isEmpty()) {
            TreeNode left = que.poll();
            TreeNode right = que.poll();
            if(left == null && right == null) {
                continue;
            }else if(left == null || right == null){
                return false;
            }
            if(left.val != right.val) return false;
            que.offer(left.left);
            que.offer(right.right);
            que.offer(left.right);
            que.offer(right.left);
        }
        return true;
    }
}
```

### 102. 二叉树的层序遍历

给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。

**示例：**

二叉树：[3,9,20,null,null,15,7]

       3
      / \
      9  20
        /  \
       15   7
       
    返回其层序遍历结果：
    [
      [3],
      [9,20],
      [15,7]
    ]
链接：https://leetcode-cn.com/problems/binary-tree-level-order-traversal

#### 解法一：递归

其实这题和上面一题的思路很像，无论是递归还是迭代的写法。

```java
class Solution {
    List<List<Integer>> lists = new ArrayList<>();
	public List<List<Integer>> levelOrder(TreeNode root) {
		levelOrderTool(root , 0);
		return lists;
	}
	void levelOrderTool(TreeNode node , int level){
		if (node==null)
			return;
		if (lists.size() <= level)
			lists.add(new ArrayList<Integer>());
		lists.get(level).add(node.val);
		levelOrderTool(node.left,level+1);
		levelOrderTool(node.right,level+1);
	}
}
```

#### 解法二：迭代

依然是用一个队列

```java
class Solution {    
	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> lists = new ArrayList<>();
        if(root == null) return lists;
        lists.add(Arrays.asList(root.val));
        Queue<TreeNode> tree = new LinkedList<>();
        if(root.left != null) tree.offer(root.left);
        if(root.right != null) tree.offer(root.right);
        while(!tree.isEmpty()) {
            //获取当前这一层有多少个节点
            int size = tree.size();
            ArrayList<Integer> tmp = new ArrayList<>();
            while(size-- > 0) {
                //用该循环一一取出当前层的所有节点
                root = tree.poll();
                //将值加入列表
                tmp.add(root.val);
                //将左右孩子加入队列
                if(root.left != null) tree.offer(root.left);
                if(root.right != null) tree.offer(root.right);
            }
            lists.add(tmp);
        }
		return lists;
	}
}
```

### 103. 二叉树的锯齿形层序遍历

给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

**示例：**

二叉树：[3,9,20,null,null,15,7]

       3
      / \
      9  20
        /  \
       15   7
       
    返回锯齿形层序遍历如下：
    [
      [3],
      [20,9],
      [15,7]
    ]

连接：https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/

#### 解法一：迭代队列

和上面一题的迭代写法基本一致，只不过需要加一个布尔值来作判断是否要从右开始，如果是的话，就要翻转一下tmp。

```java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> lists = new ArrayList<>();
        if(root == null) return lists;
        lists.add(Arrays.asList(root.val));
        Queue<TreeNode> tree = new LinkedList<>();
        if(root.left != null) tree.offer(root.left);
        if(root.right != null) tree.offer(root.right);       
        boolean leftFirst = false;
        while(!tree.isEmpty()) {
            //获取当前这一层有多少个节点
            int size = tree.size();
            ArrayList<Integer> tmp = new ArrayList<>();
            while(size-- > 0) {
                //用该循环一一取出当前层的所有节点
                root = tree.poll();
                //将值加入列表
                tmp.add(root.val);
                //将左右孩子加入队列
                if(root.left != null) tree.offer(root.left);
                if(root.right != null) tree.offer(root.right);
            }
            if(!leftFirst) {
                //如果要从右边开始就翻转数组
                Collections.reverse(tmp);
                leftFirst = true;
            }else {
                leftFirst = false;
            }
            lists.add(tmp);
        }
		return lists;
    }
}
```

#### 解法二：递归

递归步骤也基本和上一题一样，唯一不一样的地方就是对层数做了判断，如果是奇数，那么就正常直接加入即可，如果是偶数，首先判断这一层是否size=0，不等于0的话就往位置0的地方插入新数据。

```java
class Solution {
    List<List<Integer>> lists = new ArrayList<>();  
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        levelOrderTool(root, 0);      
		return lists;
    }
    void levelOrderTool(TreeNode node , int level){
		if(node==null) return;
		if(lists.size() <= level) lists.add(new ArrayList<Integer>());
		if(level%2 == 0 || lists.get(level).size() == 0){
            lists.get(level).add(node.val);
        }else {
            lists.get(level).add(0,node.val);
        }
		levelOrderTool(node.left,level+1);
		levelOrderTool(node.right,level+1);
	}
}
```

### 104. 二叉树的最大深度

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 [3,9,20,null,null,15,7]，

        3
       / \
      9  20
        /  \
       15   7
链接：https://leetcode-cn.com/problems/maximum-depth-of-binary-tree

#### 解法一：递归

相当简单，没啥好说的。

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }
}
```

#### 解法二：迭代

和上面几题的迭代写法十分雷同。

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        int depth = 1;
        Queue<TreeNode> tree = new LinkedList<>();
        if(root.left != null) tree.offer(root.left);
        if(root.right != null) tree.offer(root.right);
        while(!tree.isEmpty()) {
            depth++;
            int size = tree.size();
            while(size-- > 0) {
                root = tree.poll();
                if(root.left != null) tree.offer(root.left);
                if(root.right != null) tree.offer(root.right);
            }
        }
        return depth;
    }
}
```

### 105. 从前序与中序遍历序列构造二叉树

根据一棵树的前序遍历与中序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

```
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
20 9 15 3 7
```

返回如下的二叉树：

        3
       / \
      9  20
        /  \
       15   7
链接：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal

#### 解法一：递归

观察中序遍历，看到3的左边就是左子树，右边就是右子树。所以要先找到这个分割点，这个分割点就要靠前序遍历。创建树的时候按照前序遍历的顺序来创建，然后根据中序遍历哪边是左，哪边是右。

```java
class Solution {
    int preIndex = 0; //全局变量保存前序遍历中的指针索引
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preIndex >= preorder.length) return null;
        //创建当前节点
        int rootVal = preorder[preIndex++];
        int i = 0;
        TreeNode root = new TreeNode(rootVal);
        for(; i < inorder.length; i++) {
            //找到当前值在中序遍历中的位置
            if(inorder[i] == rootVal) break;
        }
        if(i == 0) {
            //i在中序遍历数组的最左边，代表左边无节点
            root.left = null; 
        }else {
            //如果有左子树，将中序遍历的数组截左边部分
            root.left = buildTree(preorder, Arrays.copyOfRange(inorder, 0, i));
        }     
        if(i == inorder.length - 1) {
            //i在中序遍历数组的最右边，代表右边无节点
            root.right = null;
        }else {
            //如果有右子树，将中序遍历的数组截右边部分
            root.right = buildTree(preorder, Arrays.copyOfRange(inorder, i+1, inorder.length));
        }
        return root;
    }
}
```

也可以不截断数组，只传递数组的上下界。

```java
class Solution {
    int preIndex = 0;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTree(preorder, inorder, 0, inorder.length - 1);
    }
    private TreeNode buildTree(int[] preorder, int[] inorder, int lowerBound, int upperBound) {
        if(preIndex >= preorder.length || lowerBound > upperBound) return null;
        int rootVal = preorder[preIndex++];
        TreeNode root = new TreeNode(rootVal);
        int i = lowerBound;
        for(; i <= upperBound; i++) {
            if(inorder[i] == rootVal) break;
        }
        root.left = buildTree(preorder, inorder, lowerBound, i-1);
        root.right = buildTree(preorder, inorder, i+1, upperBound);
        
        return root;
    }
}
```

#### 解法二：递归

上面解法的代码思路很明确，但是每一次都要通过遍历寻找该节点在中序数组中的位置，比较麻烦。看了高分代码如下。因为是按照前序遍历的顺序在创建节点，所以只要不是空就一定创建的是pre指针指向的数值节点。接下来的问题就是如何判断左右孩子是否为空。

因为中序数组中一定是先左 后中 再右，在创建左孩子的时候father参数传入父节点，也就是**中节点**的值，如果传入的father满足`inorder[in] == father`，就说明：没有左孩子，返回null，还要给in指针加一，此时in指针指向的应当是右子树的位置。如果不满足上面的条件，就说明有左孩子，并且in指针正指向左子树中最左边节点的位置。

然后，在创建右孩子的时候，一定是已经从左孩子的递归调用中返回过了，即in指针已经移动到了右子树的位置，即当前节点在中序遍历中右边一位的位置。father传入当前节点的父节点值，如果有右孩子，那么现在in指向的就是应当创建的右孩子的值，肯定不和这个father值相等。如果满足`inorder[in] == father`，就说明：没有右孩子，返回null，给in指针加一，指向下一棵树的左子树。

代码虽然很简洁，但是很不好想。

```java
class Solution {
    int pre = 0;
    int in = 0;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTree(preorder,inorder,Integer.MIN_VALUE - 1l);
    }
    public TreeNode buildTree(int[] preorder, int[] inorder,long father) {
        if(pre == preorder.length) return null;
        if(inorder[in] == father){
            in++;
            return null;
        }
        int rootVal = preorder[pre++];
        TreeNode root = new TreeNode(rootVal);
        root.left = buildTree(preorder,inorder,rootVal);
        root.right = buildTree(preorder,inorder,father);
        return root;
    }
}
```

#### 解法三：迭代 栈

这里用到的思想其实和解法二差不多，只不过换成了迭代的方法。

```java
class Solution { 
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder.length == 0) return null;
        int pre = 0;
        int in = 0;
        TreeNode root = new TreeNode(preorder[pre++]);
        TreeNode curRoot = root;
        Stack<TreeNode> tree = new Stack<>();
        tree.push(curRoot);
        while(pre < preorder.length) {
            if(inorder[in] == curRoot.val) {
                //创建右子树
                while(!tree.isEmpty() && tree.peek().val == inorder[in]) {//一直找到一个有右子树的节点
                    in++;
                    curRoot = tree.pop();                    
                }
                curRoot.right = new TreeNode(preorder[pre++]);
                curRoot = curRoot.right;    
            }else {
                //创建左子树
                curRoot.left = new TreeNode(preorder[pre++]);
                curRoot = curRoot.left;
            }
            tree.push(curRoot);
        }
        return root;
    }
}
```

### 106. 从中序与后序遍历序列构造二叉树

根据一棵树的中序遍历与后序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

```
中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
```

返回如下的二叉树：

    	3
       / \
      9  20
        /  \
       15   7

链接：https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal

#### 解法一：递归

和上面解法一基本类似，稍作修改即可，不同的地方在于，后序遍历是左 右 中，倒着找最上面的节点的时候是中 右 左，所以要先创建右节点。

```java
class Solution {
    int postIndex;
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        postIndex = postorder.length - 1;
        return buildTree(inorder, postorder, 0, inorder.length - 1);
    }
    private TreeNode buildTree(int[] inorder, int[] postorder, int lowerBound, int upperBound) {
        if(postIndex < 0 || lowerBound > upperBound) return null;
        int rootVal = postorder[postIndex--];
        TreeNode root = new TreeNode(rootVal);
        int i = lowerBound;
        for(; i <= upperBound; i++) {
            if(inorder[i] == rootVal) break;
        }
        root.right = buildTree(inorder, postorder, i+1, upperBound);
        root.left = buildTree(inorder, postorder, lowerBound, i-1);
        return root;
    }
}
```

那么不改变左右节点创建顺序，仍然从左开始可以吗？答案是可以的。通过中序遍历数组找到了当前节点，可以知道左子树有多少个节点，右子树有多少节点，因而可以得到下一个要创建的节点在后序数组中的位置。

```java
class Solution {
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if(inorder.length == 0) return null;
        int len = inorder.length;
        return buildTree(inorder, postorder, 0, len - 1, 0, len - 1);
    }
    public TreeNode buildTree(int[] inorder, int[] postorder, int indexInS, int indexInE, int IndexPosS, int IndexPosE) {
        if(indexInS > indexInE || IndexPosS > IndexPosE) return null;
        TreeNode root = new TreeNode(postorder[IndexPosE]);
        int newIE = 0, newPS = 0;
        for(int i = indexInS; i <= indexInE; i++) {
            if(inorder[i] == postorder[IndexPosE]) {
                newIE = i; //找到inorder数组切割索引：即本root在inorder数组中的位置
                break;
            }
        }
        newPS = IndexPosS + newIE - indexInS - 1;
        root.left = buildTree(inorder,postorder,indexInS,newIE-1,IndexPosS,newPS);
        root.right = buildTree(inorder,postorder,newIE+1,indexInE,newPS+1,IndexPosE-1);
        return root;
    }
}
```

#### 解法二：递归

和上面解法二类似，不过现在两个数组的指针都是从末尾开始，且先创建右节点再创建左节点。

```java
class Solution {
    int post;
    int in;
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        post = in = postorder.length - 1;
        return buildTree(inorder, postorder, Integer.MIN_VALUE-1l);
    }
    private TreeNode buildTree(int[] inorder, int[] postorder, long father) {
        if(post < 0) return null;
        if(inorder[in] == father) {
            in--;
            return null;
        }
        int rootVal = postorder[post--];
        TreeNode root = new TreeNode(rootVal);
        root.right = buildTree(inorder, postorder, rootVal);
        root.left = buildTree(inorder, postorder, father);      
        return root;
    }
}
```

#### 解法三：迭代 栈

和解法二的思路是一样的。

```java
class Solution {
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        int post;
        int in;
        post = in = postorder.length - 1;
        if(post < 0) return null;
        TreeNode root = new TreeNode(postorder[post--]);
        TreeNode curRoot = root;
        Stack<TreeNode> tree = new Stack<>();
        tree.push(root);
        while(post >= 0) {
            if(inorder[in] == curRoot.val) {
                //创建左子树
                while(!tree.isEmpty() && inorder[in] == tree.peek().val) {
                    curRoot = tree.pop();
                    in--;
                }
                curRoot.left = new TreeNode(postorder[post--]);
                curRoot = curRoot.left;
            }else {
                //创建右子树
                curRoot.right = new TreeNode(postorder[post--]);
                curRoot = curRoot.right;
            }
            tree.push(curRoot);
        }
        return root;
    }
}
```

### 107. 二叉树的层序遍历 II

给定一个二叉树，返回其节点值自底向上的层序遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

例如：
给定二叉树 [3,9,20,null,null,15,7],

    	3
       / \
      9  20
        /  \
       15   7

  返回其自底向上的层序遍历为：

```
[
  [15,7],
  [9,20],
  [3]
]
```

链接：https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii

#### 解法一：深度优先搜索

与普通的层序遍历不同的是，要自底向上，那么确定当前层数在返回的数组链表中的位置就很重要了。该位置很容易可以求得是`result.size() - level - 1`，得出下面的代码。

```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        result.add(new ArrayList<Integer> ());
        dfs(result,root,0);
        return result;
    }
    private void dfs(List<List<Integer>> result, TreeNode node, int level) {
        if(node == null) return;
        if(level >= result.size()) result.add(0,new ArrayList<Integer> ());
        result.get(result.size() - level - 1).add(node.val);
        dfs(result, node.left, level+1);
        dfs(result, node.right, level+1);
    }
}
```

想偷懒的话，直接用正常的层序遍历代码，再反转即可。

```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        dfs(result,root,0);
        Collections.reverse(result);
        return result;
    }
    private void dfs(List<List<Integer>> result, TreeNode node, int level) {
        if(node == null) return;
        if(level >= result.size()) result.add(new ArrayList<Integer> ());
        result.get(level).add(node.val);
        dfs(result, node.left, level+1);
        dfs(result, node.right, level+1);
    }
}
```

#### 解法二：广度优先遍历

每次只需要往数组链表的第一个插入即可。

```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;        
        Queue<TreeNode> tree = new LinkedList<>();
        tree.offer(root);
        result.add(Arrays.asList(root.val));

        while(!tree.isEmpty()) {
            int size = tree.size();
            ArrayList<Integer> tmp = new ArrayList<>();
            while(size-- > 0) {
                root = tree.poll();                
                if(root.left != null) {
                    tree.offer(root.left);
                    tmp.add(root.left.val);
                }
                if(root.right != null) {
                    tree.offer(root.right);
                    tmp.add(root.right.val);
                }                
            }
            if(tmp.size() > 0) result.add(0,tmp);
        }
        return result;
    }
}
```

同样的，也可以用`Collections.reverse(result);`偷懒

```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        que.offer(root);
        while(!que.isEmpty()) {
            int size = que.size();
            List<Integer> seg = new ArrayList<>();
            for(int i = 0; i < size; i++) {
                TreeNode node = que.poll();
                seg.add(node.val);
                if(node.left != null) que.offer(node.left);
                if(node.right != null) que.offer(node.right);
            }
            result.add(seg);
        }
        Collections.reverse(result);
        return result;
    }
}
```

### 108. 将有序数组转换为二叉搜索树

给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。

**示例 1：**

![](https://assets.leetcode.com/uploads/2021/02/18/btree1.jpg)

```
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：
```

![](https://assets.leetcode.com/uploads/2021/02/18/btree2.jpg)

**示例 2：**

![](https://assets.leetcode.com/uploads/2021/02/18/btree.jpg)

```
输入：nums = [1,3]
输出：[3,1]
解释：[1,3] 和 [3,1] 都是高度平衡二叉搜索树。
```

链接：https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree

#### 解法一：递归

要求高度差不超过1，那么就取中间的那个值为根节点即可，然后按照同样的理论一路往左或往右创建子树。

```java
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }
    private TreeNode sortedArrayToBST(int[] nums, int left, int right) {
        if(left > right) return null;
        int mid = left + (right - left)/2; //mid = (start + end) >>> 1;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, left, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, right);
        return root;
    }
}
```

#### 解法二：迭代 栈

看[这里](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-24/)，通过一个栈模拟递归。递归时候需要的left和right参数被和root一起封装成了一个类。然后先一直产生左节点并压入栈，等不满足生成左节点时再弹栈，生成其右节点。

```java
class MyTreeNode {
    TreeNode root;
    int start;
    int end;

    MyTreeNode(TreeNode r, int s, int e) {
        this.root = r;
        this.start = s;
        this.end = e;
    }
}
public TreeNode sortedArrayToBST(int[] nums) {
    if (nums.length == 0) {
        return null;
    }
    Stack<MyTreeNode> rootStack = new Stack<>();
    int start = 0;
    int end = nums.length;
    int mid = (start + end) >>> 1;
    TreeNode root = new TreeNode(nums[mid]);
    TreeNode curRoot = root;
    rootStack.push(new MyTreeNode(root, start, end));
    while (end - start > 1 || !rootStack.isEmpty()) {
        //考虑左子树
        while (end - start > 1) {
            mid = (start + end) >>> 1; //当前根节点
            end = mid;//左子树的结尾
            mid = (start + end) >>> 1;//左子树的中点
            curRoot.left = new TreeNode(nums[mid]);
            curRoot = curRoot.left;
            rootStack.push(new MyTreeNode(curRoot, start, end));
        }
        //出栈考虑右子树
        MyTreeNode myNode = rootStack.pop();
        //当前作为根节点的 start end 以及 mid
        start = myNode.start;
        end = myNode.end;
        mid = (start + end) >>> 1;
        start = mid + 1; //右子树的 start
        curRoot = myNode.root; //当前根节点
        if (start < end) { //判断当前范围内是否有数
            mid = (start + end) >>> 1; //右子树的 mid
            curRoot.right = new TreeNode(nums[mid]);
            curRoot = curRoot.right;
            rootStack.push(new MyTreeNode(curRoot, start, end));
        }

    }

    return root;
}
```

同样也可以用队列+迭代的方法来写层序遍历的生成。太麻烦了不想写了，偷个懒。

### 109. 有序链表转换二叉搜索树

给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

**示例:**

    给定的有序链表： [-10, -3, 0, 5, 9],
    
    一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：  
          0
         / \
        -3   9
       /   /
     -10  5
链接：https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree

#### 解法一：利用辅助数组

108题给出的是一个有序数组，可以通过把链表转换成数组来解决这题。

```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        ArrayList<Integer> array = new ArrayList<>();
        while(head != null) {
            array.add(head.val);
            head = head.next;
        }
        return sortedArrayToBST(array, 0, array.size() - 1);
    }
    private TreeNode sortedArrayToBST(ArrayList<Integer> nums, int left, int right) {
        if(left > right) return null;
        int mid = left + (right - left)/2; //mid = (start + end) >>> 1;
        TreeNode root = new TreeNode(nums.get(mid));
        root.left = sortedArrayToBST(nums, left, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, right);
        return root;
    }
}
```

#### 解法二：不用数组辅助

如果不利用数组的辅助，这题怎么写？

直接的做法是先遍历链表，找到整条链表的大小，然后每次求出mid，遍历链表找到这个值。不过这样做显然时间复杂度很高。

```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        ListNode p = head;
        int size = 0;
        while(p != null) {
            p = p.next;
            size++;
        }
        return sortedListToBST(head, 0, size - 1);
    }
    private TreeNode sortedListToBST(ListNode head, int left, int right) {
        if(left > right) return null;
        int mid = left + (right - left)/2; //mid = (start + end) >>> 1;
        ListNode cur = head;
        for(int i = 0; i < mid; i++) {
            cur = cur.next;
        }
        TreeNode root = new TreeNode(cur.val);
        root.left = sortedListToBST(head, left, mid - 1);
        root.right = sortedListToBST(head, mid + 1, right);
        return root;
    }
}
```

#### 解法三：快慢指针

可以通过快慢指针来寻找中间节点，快指针每次走两步，慢指针每次走一步，当快指针到达链表尾的时候，慢指针的位置就是中间节点。并且也不再传入left和right指针作为形参，而是传入头尾节点。

```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        return sortedListToBST(head, null);
    }
    private TreeNode sortedListToBST(ListNode head, ListNode tail) {
        if(head == tail) return null;
        ListNode slow = head, fast = head;
        while(fast != tail && fast.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);
        root.left = sortedListToBST(head, slow);
        root.right = sortedListToBST(slow.next, tail);
        return root;
    }
}
```

#### 解法四：中序遍历创建

上面每次都要寻找中间节点，时间复杂度还是有点高的。

下面这种写法感觉很牛。实质上是利用中序遍历的方法来创建树。仔细想想，其实给的这个链表不就是按照中序遍历的顺序摆放的嘛。重点是如何确定中间节点的位置，可以根据left和right这两个参数。

```java
class Solution {
    ListNode cur = null;
    public TreeNode sortedListToBST(ListNode head) {
        if(head == null) return null;
        cur = head;
        int size = 0;
        while(head != null) {
            size++;
            head = head.next;
        }
        return sortedListToBST(0,size-1);
    }
    private TreeNode sortedListToBST(int left, int right) {
        if(left > right) return null;
        int mid = (left + right) >>> 1;
        TreeNode leftNode = sortedListToBST(left,mid-1);
        TreeNode root = new TreeNode(cur.val);
        root.left = leftNode;
        cur = cur.next;
        TreeNode rightNode = sortedListToBST(mid+1,right);
        root.right = rightNode;
        return root;
    }
}
```

### 110. 平衡二叉树

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/10/06/balance_1.jpg)

```
输入：root = [3,9,20,null,null,15,7]
输出：true
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2020/10/06/balance_2.jpg)

```
输入：root = [1,2,2,3,3,null,null,4,4]
输出：false
```

**示例 3：**

```
输入：root = []
输出：true
```

链接：https://leetcode-cn.com/problems/balanced-binary-tree

#### 解法：int也能判断true or false

最直接的想法就是一个返回值是boolean的递归函数，判断，左右子树是否为平衡树并且判断当前树是否为平衡树。树的高度通过另一个返回值是int的递归函数取得。但是这样写会发现，做了很多重复性的工作。以示例2的图为例，在求2的高度的时候，会重新执行取得树高度的递归函数，但是实际上只需要在以3为根节点的高度上加一就行了。

那么怎么样写能避免既得到高度又得到是否为平衡树的布尔值呢？

在求每个节点的高度的时候，是要取左右树高度的最大值再+1，如果左右树高度差已经不满足要求，就将返回值置位-1，这就代表，出现了不合法的二叉树。

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return (findHeight(root) > -1);
    }
    private int findHeight(TreeNode node) {
        if(node == null) return 0;
        int h1 = findHeight(node.left);
        if(h1 == -1) return -1;
        h1++;
        int h2 = findHeight(node.right);
        if(h2 == -1) return -1;
        h2++;
        
        if(Math.abs(h1 - h2) > 1) return -1;
        return Math.max(h1,h2);
    }
}
```

这题虽然不难，但是想要写出最优解很难。难点就在于将int数值看成一个可以传递真假值的数。

### 111. 二叉树的最小深度

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明：叶子节点是指没有子节点的节点。

**示例 1：**

```
输入：root = [3,9,20,null,null,15,7]
输出：2
```

**示例 2：**

```
输入：root = [2,null,3,null,4,null,5,null,6]
输出：5
```

链接：https://leetcode-cn.com/problems/minimum-depth-of-binary-tree。

#### 解法一：深度优先搜索

这题不是把搜索最大深度代码中的max换成min函数即可。因为题目要求，是从根节点，到一个没有子节点的节点的深度，如果直接换成min，遇到一个只有右子节点的树会返回1，但这并不是真正的答案。

要找到左右子树的高度之后再做进一步判断，看是否有一边高度为0。

```java
class Solution {
    public int minDepth(TreeNode root) {
        if(root == null) return 0;
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if(left == 0 && right == 0) return 1;
        if(left == 0) return 1 + right;
        if(right == 0) return 1 + left;
        return 1 + Math.min(left,right);  
    }
}
```

看到一种很简练的写法，不过道理都是一样的：

```java
class Solution {
    public int minDepth(TreeNode root) {
        if(root == null) return 0;
        if(root.left != null && root.right != null) {
            return 1 + Math.min(minDepth(root.left),minDepth(root.right));  
        }
        return 1 + Math.max(minDepth(root.left),minDepth(root.right));       
    }
}
```

#### 解法二：广度优先搜索

用层间遍历的方法，当发现某一个节点的左右孩子都是空就立即返回深度。

```java
class Solution {
    public int minDepth(TreeNode root) {
        if(root == null) return 0;
        int depth = 1;
        Queue<TreeNode> tree = new LinkedList<>();
        tree.offer(root);
        while(!tree.isEmpty()) {
            int size = tree.size();
            while(size-- > 0) {
                root = tree.poll();
                if(root.left == null && root.right == null) return depth;
                if(root.left != null) tree.offer(root.left);
                if(root.right != null) tree.offer(root.right);
            }
            depth++;
        }
        return depth;
    }
}
```

### 112. 路径总和

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。

叶子节点 是指没有子节点的节点。

**示例 1：**

![](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

```
输入：root = [1,2,3], targetSum = 5
输出：false
```

**示例 3：**

```
输入：root = [1,2], targetSum = 0
输出：false
```

链接：https://leetcode-cn.com/problems/path-sum

#### 解法一：递归

```java
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root == null) return false;
        if(targetSum == root.val && root.left == null && root.right == null) return true;           
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }
}
```

#### 解法二：迭代

要用到两个队列，一个保存树的节点，一个保存对应节点获得的加和。

```java
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root == null) return false;
        if(targetSum == root.val && root.left == null && root.right == null) return true; 
        Queue<TreeNode> tree = new LinkedList<>();
        Queue<Integer> sum = new LinkedList<>();
        tree.offer(root);
        sum.offer(root.val);  
        while(!tree.isEmpty()) {
            int size = tree.size();
            while(size-- > 0) {
                root = tree.poll();
                int s = sum.poll();
                if(s == targetSum && root.left == null && root.right == null) return true;
                if(root.left != null) {
                    tree.offer(root.left);
                    sum.offer(root.left.val + s);
                }
                if(root.right != null) {
                    tree.offer(root.right);
                    sum.offer(root.right.val + s);
                }
            }
        }       
        return false;
    }
}
```

### 113. 路径总和 II

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。 

**示例 1：**

![](https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg)

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

```
输入：root = [1,2,3], targetSum = 5
输出：[]
```

**示例 3：**

```
输入：root = [1,2], targetSum = 0
输出：[]
```

链接：https://leetcode-cn.com/problems/path-sum-ii

#### 解法：回溯法

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<Integer> path = new ArrayList<>();
        backTracing(root,targetSum,path);
        return result;
    }
    private void backTracing(TreeNode root, int targetSum, List<Integer> path) {
        if(root == null) return;
        path.add(root.val);
        if(targetSum == root.val && root.left == null && root.right == null) {
            result.add(new ArrayList<>(path));
            path.remove(path.size() - 1);
            return;
        }
        backTracing(root.left, targetSum - root.val,path);
        backTracing(root.right, targetSum - root.val,path);
        path.remove(path.size() - 1);
    }
}
```

### 114. 二叉树展开为链表

给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

**示例 1：**

![](https://assets.leetcode.com/uploads/2021/01/14/flaten.jpg)

```
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
```

**示例 2：**

```
输入：root = []
输出：[]
```

**示例 3：**

```
输入：root = [0]
输出：[0]
```

链接：https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list

#### 解法一：前序遍历存储节点

通过前序遍历将每个节点按照顺序放入数组中，然后从前到后重新创建

```java
class Solution {
    public void flatten(TreeNode root) {        
        List<TreeNode> list = new ArrayList<>();
        flattenHelp(root,list);
        if(list.size() == 0) return;
        //创建一个伪头节点
        TreeNode cur = new TreeNode(0);
        for(TreeNode node : list) {
            cur.right = node;
            cur = cur.right;
            cur.left = null;
        }
    }
    private void flattenHelp(TreeNode root, List<TreeNode> list) {
        if(root == null) return;
        //前序遍历
        list.add(root);
        flattenHelp(root.left,list);
        flattenHelp(root.right,list);
    }
}
```

#### 解法二：递归

上面的方法需要单独空间存放各个节点，考虑一种不需要另外数组空间的写法。

先确定操作的顺序，把右边的树接到左边树的**最后一个节点**后面，再把整棵左边的树接到根节点的右边。因为在根节点的时候，左右节点都可以取得，我们缺少的是尾节点，所以就将它作为递归函数返回的参数。当左右两棵树合并以后，传回的是原右边树的尾节点。

```java
class Solution {
    public void flatten(TreeNode root) {
        flattenHelp(root);
    }
    private TreeNode flattenHelp(TreeNode root) {
        if(root == null) return null;
        TreeNode tailLeft = flattenHelp(root.left);//左边尾节点
        TreeNode tailRight = flattenHelp(root.right);//右边尾节点
        //如果左边尾节点是null,代表当前根节点只有右子树
        //如果在这个情况下右边尾节点也是null,代表当前根节点就是尾节点
        if(tailLeft == null) return tailRight == null ? root : tailRight;
        //如果右边尾节点是null代表当前根节点只有左子树
        //将左树变为右子树
        //传回的尾节点应当是左边的尾节点
        if(tailRight == null) {
            root.right = root.left;
            root.left = null;
            return tailLeft;
        }
        TreeNode tmp = root.right;
        tailLeft.right = tmp;
        root.right = root.left;
        root.left = null;
        return tailRight;
    }
}
```

### 115. 不同的子序列

给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。

字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）

题目数据保证答案符合 32 位带符号整数范围。

**示例 1：**

```
输入：s = "rabbbit", t = "rabbit"
输出：3
解释：
如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
(上箭头符号 ^ 表示选取的字母)
rabbbit
^^^^ ^^
rabbbit
^^ ^^^^
rabbbit
^^^ ^^^
```

**示例 2：**

```
输入：s = "babgbag", t = "bag"
输出：5
解释：
如下图所示, 有 5 种可以从 s 中得到 "bag" 的方案。 
(上箭头符号 ^ 表示选取的字母)
babgbag
^^ ^
babgbag
^^    ^
babgbag
^    ^^
babgbag
  ^  ^^
babgbag
    ^^^
```

链接：https://leetcode-cn.com/problems/distinct-subsequences

#### 解法：动态规划

确定`dp[i][j]`的含义是字符串s从开始到位置`i`与字符串t从开始到位置`j`相等，有多少种。

当字符串s位置`i`字母与字符串t位置`j`字母不相同的时候，代表只能删除位置`i`字母，所以`dp[i][j] = dp[i-1][j]`。当相同的时候，可以保留当前位置的字母，那么就要看`dp[i-1][j-1]`是多少，也可以删除当前位置字母，这就要看`dp[i-1][j]`有多少种。

```java
class Solution {
    public int numDistinct(String s, String t) {
        int len1 = s.length(), len2 = t.length();
        if (len1<len2) return 0;
        if (len2 == 0) return 1;
        char[] S = s.toCharArray();
        char[] T = t.toCharArray();
        int[][] dp = new int[len1+1][len2+1];
        for(int i = 0; i <= len1; i++) {
            dp[i][0] = 1;
        }
        for(int i = 1; i <= len1; i++) {
            for(int j = 1; j <= i && j <= len2; j++) {
                if(S[i-1] != T[j-1]) {
                    dp[i][j] = dp[i-1][j];
                }else {
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-1];
                }
            }
        }
        return dp[len1][len2];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（m*n）

考虑降维数组来实现这题，通过观察可以发现`dp[i][j]`只取决于`dp[i-1][j-1]`和`dp[i-1][j]`，如果只用一维，想要保留这两个值不被覆盖，整个遍历过程就得从一维dp数组的最尾端往前。

```java
class Solution {
    public int numDistinct(String s, String t) {
        int len1 = s.length(), len2 = t.length();
        if (len1<len2) return 0;
        if (len2 == 0) return 1;
        char[] S = s.toCharArray();
        char[] T = t.toCharArray();
        int[] dp = new int[len2+1];
        dp[0] = 1;
        for(int i = 1; i <= len1; i++) {
            for(int j = len2; j >= 1; j--) {
                if(j > i) continue;
                if(S[i-1] == T[j-1]) {
                    dp[j] += dp[j-1];
                }
            }
        }
        return dp[len2];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（n）

### 121. 买卖股票的最佳时机

给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

**示例 1：**

```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

**示例 2：**

```
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
```


链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock

#### 解法一：动态规划

每一天有两种状态：已经买入或已经卖出。`dp[i][0]`表示第`i`天已经买入需要花费的最少的钱，`dp[i][1]`表示第`i`天已经卖出能获得的最大收益。

```java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }
        int[][] dp = new int[prices.length + 1][2];
        dp[0][0] = Integer.MIN_VALUE;
        for(int i = 0; i < prices.length; i++) {
            //当前已经是买入状态需要花的最少价钱
            dp[i+1][0] = Math.max(-prices[i],dp[i][0]);
            //当前如果是已经卖出状态所能获得的最大收益
            dp[i+1][1] = Math.max(prices[i] + dp[i][0],dp[i][1]);
        }
        return dp[prices.length][1];
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

但是上面的代码，说白了其实就是要找一个需要花费最少的那天买入，然后找一个可以卖最多钱的收益。并不需要一个数组做动态规划写，可以直接只用两个变量。这就有了下面解法二的贪心算法。

或者还有另一种动态规划的写法，`dp[i]`代表第i天卖出（不是卖出状态，而是必须在这一天卖出）所获得的最大收益，然后再用一个变量保存一个最大值。

```java
class Solution {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        int[] dp = new int[len];
        int max = 0;
        for(int i = 1; i < len; i++) {
            dp[i] = prices[i] - prices[i-1] + (dp[i-1] <= 0 ? 0 : dp[i-1]);
            max = Math.max(max, dp[i]);
        }
        return max;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

#### 解法二：贪心算法

用两个变量，一个存储遍历到的最小值min，一个存储之前计算到的可以取得的最大利润stock。

从前往后遍历，如果min大于当前值，代表min需要被更新。如果min小于当前值，可以计算一下stock，如果计算得到的新stock并没有大于之前的stock那么就不需要更新，否则更新。

```java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }
        int min = Integer.MAX_VALUE;
        int stock = 0;
        for (int i = 0; i < prices.length; i++) {
            if (min > prices[i]) {
                min = prices[i];
            } else {
                stock = Math.max(prices[i] - min, stock);
            }
        }
        return stock;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 122. 买卖股票的最佳时机 II

给定一个数组 prices ，其中 prices[i] 是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例 1:**

```
输入: prices = [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
```

**示例 2:**

```
输入: prices = [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```

**示例 3:**

```
输入: prices = [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii

#### 解法一：贪心算法

只要遇到比前一天少的价格就把前一天的抛出然后立刻买入这一天。

```java
class Solution {
    public int maxProfit(int[] prices) {
        //buy是买入花费的钱，profit是取得的最大收益，cur是当前股票价格
        int buy = prices[0], profit = 0, cur = 0;
        for(int i = 1; i < prices.length; i++) {
            cur = prices[i];
            //如果今天的股票价格比前一天低
            if(cur < prices[i-1]) {  
                //如果前一天的股票价格比买入花费的要多，就将这一笔交易能获得的钱计算归入收益
                if(prices[i-1] > buy) profit += prices[i-1] - buy;
                //记录下今天买入需要花费的钱
                buy = cur;      
            }
        }
        //最后一天需要判断，防止还未抛出
        if(cur > buy) profit += cur - buy;
        return profit;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法二：动态规划

和上面一题用到的动态规划思想本质相同，每一天有两种状态，要么是已经买入状态，要么是已经卖出的状态。

```java
class Solution {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        int[][] dp = new int[len][2];
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for(int i = 1; i < len; i++) {
            //持有股票所能得到的最大收益
            //要么是前一天持有股票的收益
            //要么是前一天不持有股票然后今天买入的收益
            //其实本质是在计算需要花费的最少金钱
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] - prices[i]);
            //不持有股票所能得到的最大收益
            //要么是前一天不持有股票的收益
            //要么是前一天持有股票然后今天卖出去的收益
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] + prices[i]);
        }
        return dp[len-1][1];//最后一天一定要卖出股票
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

### 123. 买卖股票的最佳时机 III

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例 1:**

```
输入：prices = [3,3,5,0,0,3,1,4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
```

**示例 2：**

```
输入：prices = [1,2,3,4,5]
输出：4
解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。   
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。   
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```

**示例 3：**

```
输入：prices = [7,6,4,3,1] 
输出：0 
解释：在这个情况下, 没有交易完成, 所以最大利润为 0。
```

**示例 4：**

```
输入：prices = [1]
输出：0
```

链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii

#### 解法：动态规划

这题比较复杂，应该不能再用贪心算法解决了。结合上面几题的动态规划写法，发现共性都是判断某一天有几种状态。同样的，这一题要求只有最多两次交易，那么每一天就有五种状态：什么都还没有操作；第一次买入状态；第一次卖出状态；第二次买入状态；第二次卖出状态。

```java
class Solution {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        if(len == 1) return 0;
        // 状态0：不操作
        // 状态1：第一次买入(可以是当天买入也可以表示之前买入后没有其他操作)
        // 状态2：第一次卖出(可以是当天卖出也可以表示之前卖出后没有其他操作)
        // 状态3：第二次买入
        // 状态4：第二次卖出
        int[][] dp = new int[len][5];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        dp[0][3] = -prices[0];
        for(int i = 1; i < len; i++) {
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] - prices[i]);
            dp[i][2] = Math.max(dp[i-1][2], dp[i-1][1] + prices[i]);
            dp[i][3] = Math.max(dp[i-1][3], dp[i-1][2] - prices[i]);
            dp[i][4] = Math.max(dp[i-1][4], dp[i-1][3] + prices[i]);
        }
        return dp[len-1][4];
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

可以降维只用一维数组完成。

```java
class Solution {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        if(len == 1) return 0;
        // 状态0：不操作
        // 状态1：第一次买入(可以是当天买入也可以表示之前买入后没有其他操作)
        // 状态2：第一次卖出(可以是当天卖出也可以表示之前卖出后没有其他操作)
        // 状态3：第二次买入
        // 状态4：第二次卖出
        int[] dp = new int[5];
        dp[0] = 0;
        dp[1] = -prices[0];
        dp[3] = -prices[0];
        for(int i = 1; i < len; i++) {
            dp[1] = Math.max(dp[1], dp[0] - prices[i]);
            dp[2] = Math.max(dp[2], dp[1] + prices[i]);
            dp[3] = Math.max(dp[3], dp[2] - prices[i]);
            dp[4] = Math.max(dp[4], dp[3] + prices[i]);
        }
        return dp[4];
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

这一题虽然是难题，但是其实只要想清楚每一天有哪几种状态就好解决了。

### 128. 最长连续序列

给定一个未排序的整数数组`nums`，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

进阶：你可以设计并实现时间复杂度为 O(n) 的解决方案吗？

**示例 1：**

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

**示例 2：**

```
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```

链接：https://leetcode-cn.com/problems/longest-consecutive-sequence

#### 解法一：

如果不管进阶的要求，直接用排序做的思路是很清晰的。

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        Arrays.sort(nums);
        int ans = 0;
        for(int i = 0,j; i < nums.length; i=j) {
            int length = 1;
            for(j = i+1; j < nums.length && nums[j] - nums[j - 1] <= 1; j++) {
                if(nums[j] == nums[j-1] + 1) length++;
            }
            ans = Math.max(ans,length);
        }
        return ans;
    }
}
```

时间复杂度：O（n*logn）

空间复杂度：O（1）

#### 解法二：哈希表

先是很容易想到用一个HashSet将所有的数都存储进去，但是到了查找这一步就遇到麻烦了。如果考虑枚举数组中的每个数 x，考虑以其为起点，不断尝试匹配 x+1, x+2,⋯ 是否存在，假设最长匹配到了 x+y。如果在数组中后面又遇到了一个数x-1，又会以它为起点重新开始搜索。这样整个时间复杂度会高到O（n^2^）。

[官方题解](https://leetcode-cn.com/problems/longest-consecutive-sequence/solution/zui-chang-lian-xu-xu-lie-by-leetcode-solution/)可以先查看是否有x-1，如果没有再开始搜索，如果有就先不以该数为起点继续往下搜索。

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for(int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }
        int ans = 0;
        for(int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if(!set.contains(num - 1)) {
                int count = 1;
                while(set.contains(++num)) {
                    count++;
                }
                ans = Math.max(ans,count);
            }
        }
        return ans;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

虽然时间复杂度是O（n），但是实际上调用contains函数（实现本质是暴力搜索），应该也不是很快。

### 130. 被围绕的区域

给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。

**示例 1：**

![](https://assets.leetcode.com/uploads/2021/02/19/xogrid.jpg)

```
输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
```

**示例 2：**

```
输入：board = [["X"]]
输出：[["X"]]
```


链接：https://leetcode-cn.com/problems/surrounded-regions

#### 解法一：深度遍历搜索

在考虑的前期有一个误区，总觉得要排除掉四条边对中间的O区域进行递归，但是怎么写都不对。才发现思路有个大问题，其实应该先对四条边上的O进行递归，把所有与边界连接的O区域先标记掉，再对内部区域进行遍历，此时搜索到的连通O区域就可以修改值了。

```java
class Solution {
    boolean[][] used;
    int m,n;
    int[] dir = {1, 0, -1, 0, 1};
    public void solve(char[][] board) {
        m = board.length;
        n = board[0].length;
        used = new boolean[m][n];
        for(int i = 0; i < m; i++) {
            if(board[i][0] == 'O' && !used[i][0]) {
                used[i][0] = true;
                dfs(board,i,0,true); //对左边界进行搜索
            }
            if(board[i][n-1] == 'O' && !used[i][n-1]) {
                used[i][n-1] = true;
                dfs(board,i,n-1,true); //对右边界进行搜索
            }     
        }
        for(int i = 1; i < n-1; i++) {
            if(board[0][i] == 'O' && !used[0][i]) {
                used[0][i] = true;
                dfs(board,0,i,true); //对上边界进行搜索
            }
            if(board[m-1][i] == 'O' && !used[m-1][i]) {
                used[m-1][i] = true;
                dfs(board,m-1,i,true); //对下边界进行搜索
            }
        }
        for(int i = 1; i < m-1; i++) {
            for(int j = 1; j < n-1; j++) {
                if(board[i][j] == 'O' && !used[i][j]) {
                    board[i][j] = 'X';
                    dfs(board,i,j,false);
                }
            }
        }

    }
    private void dfs(char[][] board, int i, int j, boolean isBorder) {
        for(int s = 0; s < 4; s++) {
            int row = i+dir[s], col = j+dir[s+1];
            if(row>=0 && row<m && col>=0 && col<n) {
                if(board[row][col] == 'O' && !used[row][col]){
                    if(isBorder){
                        used[row][col] = true;
                        dfs(board,row,col,true);
                    }else{
                        board[row][col] = 'X';
                        dfs(board,row,col,false);
                    }
                }
            }
        }
    }
}
```

还看到了一种写法，是只对四条边界上遇到的O调用递归，通过递归将连通的区域全部修改为另一个值'-'，然后再对内部区域进行遍历，遇到O直接修改为'X'，因为这些区域一定是不与边界连通的，所以一定要被修改，遇到'-'改回'O'。

```java
class Solution {
	public void solve(char[][] board){
		int row = board.length;
		int column = board[0].length;
		for (int i = 0; i < row; i++){
			if(board[i][column - 1] == 'O'){
				markO(board,i,column - 1,row - 1, column - 1);
			}
			if(board[i][0] == 'O'){
				markO(board,i,0,row - 1, column - 1);
			}
		}
		for (int j = 0; j < column; j++){
			if(board[0][j] == 'O'){
				markO(board,0,j,row-1,column-1);
			}
			if(board[row - 1][j] == 'O'){
				markO(board,row - 1,j,row-1,column - 1);
			}
		}
		for (int i = 0; i <row; i++){
			for (int j = 0; j < column; j++){
				if(board[i][j] == 'O'){
					board[i][j] = 'X';
				}else if(board[i][j] == '-'){
					board[i][j] = 'O';
				}
			}
		}
	}

	public void markO(char[][] board, int i, int j, int row, int column){
		board[i][j] = '-';
		//down
		if(i < row && board[i + 1][j] == 'O'){
			markO(board,i+1,j,row,column);
		}
		//right
		if(j < column && board[i][j+1] == 'O'){
			markO(board,i,j+1,row,column);
		}
		//up
		if(i > 0 && board[i - 1][j] == 'O'){
			markO(board,i-1,j,row,column);
		}
		//left
		if(j > 0 && board[i][j - 1] == 'O'){
			markO(board,i,j-1,row,column);
		}
	}
}
```

这题的重点是要从边界突破，也就是从所谓的特殊情况开始下手突破。

#### 解法二：并查集

> 并查集的思想就是，同一个连通区域内的所有点的根节点是同一个。将每个点映射成一个数字。先假设每个点的根节点就是他们自己，然后我们以此输入连通的点对，然后将其中一个点的根节点赋成另一个节点的根节点，这样这两个点所在连通区域又相互连通了。
> 并查集的主要操作有：
>
> find(int m)：这是并查集的基本操作，查找 mm 的根节点。
>
> isConnected(int m,int n)：判断 m，nm，n 两个点是否在一个连通区域。
>
> union(int m,int n):合并 m，nm，n 两个点所在的连通区域。

作者：Ac_pipe
链接：https://leetcode-cn.com/problems/surrounded-regions/solution/bfsdi-gui-dfsfei-di-gui-dfsbing-cha-ji-by-ac_pipe/
来源：力扣（LeetCode）

```java
public class Solution {
    int rows, cols;

    public void solve(char[][] board) {
        if(board == null || board.length == 0) return;

        rows = board.length;
        cols = board[0].length;

        //多申请一个空间
        UnionFind uf = new UnionFind(rows * cols + 1);
        //所有边界的 O 节点都和 dummy 节点合并
        int dummyNode = rows * cols;

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(board[i][j] == 'O') {
                    //当前节点在边界就和 dummy 合并
                    if(i == 0 || i == rows-1 || j == 0 || j == cols-1) {
                        uf.union( dummyNode,node(i,j));
                    }
                    else {
                        //将上下左右的 O 节点和当前节点合并
                        if(board[i-1][j] == 'O')  uf.union(node(i,j), node(i-1,j));
                        if(board[i+1][j] == 'O')  uf.union(node(i,j), node(i+1,j));
                        if(board[i][j-1] == 'O')  uf.union(node(i,j), node(i, j-1));
                        if( board[i][j+1] == 'O')  uf.union(node(i,j), node(i, j+1));
                    }
                }
            }
        }

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                //判断是否和 dummy 节点是一类
                if(uf.isConnected(node(i,j), dummyNode)) {
                    board[i][j] = 'O';
                }
                else {
                    board[i][j] = 'X';
                }
            }
        }
    }

    int node(int i, int j) {
        return i * cols + j;
    }
}

class UnionFind {
    int [] parents;
    public UnionFind(int totalNodes) {
        parents = new int[totalNodes];
        for(int i = 0; i < totalNodes; i++) {
            parents[i] = i;
        }
    }

    void union(int node1, int node2) {
        int root1 = find(node1);
        int root2 = find(node2);
        if(root1 != root2) {
            parents[root2] = root1;
        }
    }

    int find(int node) {
        while(parents[node] != node) {
            parents[node] = parents[parents[node]];
            node = parents[node];
        }
        return node;
    }

    boolean isConnected(int node1, int node2) {
        return find(node1) == find(node2);
    }
}

```

### 131. 分割回文串

给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。

回文串 是正着读和反着读都一样的字符串。

**示例 1：**

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

**示例 2：**

```
输入：s = "a"
输出：[["a"]]
```

链接：https://leetcode-cn.com/problems/palindrome-partitioning

#### 解法一：回溯法

添加元素，判断是否为回文，是就进行下一层递归，返回后删除元素。

```java
class Solution {
    List<List<String>> res;
    List<String> path;
    int len;
    char[] arr;
    public List<List<String>> partition(String s) {
        //初始化
        this.res = new ArrayList<>();
        this.path = new ArrayList<>();
        this.len = s.length();
        this.arr = s.toCharArray();
        backtracing(0);
        return res;
    }
    //index 为起始位置
    public void backtracing(int index){
        if(index == len){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i = index; i < len; i++){
            if(isPalindrome(index, i)){
                path.add(new String(arr, index, i + 1 - index));
                backtracing(i + 1);
                path.remove(path.size() - 1);
            }
        }
    }
    //判断是否是回文
    public boolean isPalindrome(int left, int right){
        while(left < right){
            if(arr[left++] != arr[right--]) return false;
        }
        return true;
    }
}
```

#### 解法二：分治

将一个字符串从头开始往后分割，递归调用自身函数处理后面的字符串。

```java
class Solution {
    public List<List<String>> partition(String s) {
        int len = s.length();
        List<List<String>> res = new ArrayList<>();
        if(len == 0) {
            List<String> list = new ArrayList<>();
            res.add(list);
            return res;
        }
        
        char[] chars = s.toCharArray();
        StringBuilder path = new StringBuilder();
        int i = 0;
        while(i < len) {
            path.append(chars[i]);
            if(isPalindrome(s,0,i)) {
                String tmp = path.toString();
                for(List<String> list :partition(s.substring(i+1,len))) {
                    list.add(0,tmp);
                    res.add(list);
                }
            }
            i++;
        }
        return res;
    }
    //判断是否是回文
    public boolean isPalindrome(String s, int left, int right){
        while(left < right){
            if(s.charAt(left++) != s.charAt(right--)) return false;
        }
        return true;
    }
}
```

但这之中会做很多重复操作，原因很好想：会重复判断处理后面的字符串。那可以用一个数组`dp[i][j]`保存从`i`位置到`j`位置是否为回文。

```java
class Solution {
    boolean[][] dp;
    int len;
    public List<List<String>> partition(String s) {
        len = s.length();
        dp = new boolean[len][len];
        for(int i = 1; i <= len; i++) {
            for(int j = 0; j+i-1 < len; j++) {
                if(i == 1) {
                    dp[j][j] = true;
                }else {
                    if(s.charAt(j) == s.charAt(j+i-1)) dp[j][j+i-1] = (i == 2 || dp[j+1][j+i-2]);
                }
            }
        }
        return partitionHelp(s,0);
    }
    private List<List<String>> partitionHelp(String s, int start) {
        List<List<String>> res = new ArrayList<>();
        if(len == start) {
            List<String> list = new ArrayList<>();
            res.add(list);
            return res;
        }        
        char[] chars = s.toCharArray();
        StringBuilder path = new StringBuilder();
        int i = start;
        while(i < len) {
            path.append(s.charAt(i));
            if(dp[start][i]) {
                String tmp = path.toString();
                for(List<String> list :partitionHelp(s,i+1)) {
                    list.add(0,tmp);
                    res.add(list);
                }
            }
            i++;
        }
        return res;
    }
    //判断是否是回文
    public boolean isPalindrome(String s, int left, int right){
        while(left < right){
            if(s.charAt(left++) != s.charAt(right--)) return false;
        }
        return true;
    }
}
```

用动态规划思想做回文判断是一个重点。

### 132*. 分割回文串 II

给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文。

返回符合要求的 最少分割次数 。

**示例 1：**

```
输入：s = "aab"
输出：1
解释：只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
```

**示例 2：**

```
输入：s = "a"
输出：0
```

**示例 3：**

```
输入：s = "ab"
输出：1
```

链接：https://leetcode-cn.com/problems/palindrome-partitioning-ii

#### 解法一：分治

计算一个字符串能分割几次，可以通过将字符串分解成两部分，分别递归求可以分几次得出。并沿用上一题的思想，用一个布尔数组来记录是否为回文串。

得出下面代码

```java
class Solution {
    boolean[][] isPld;
    public int minCut(String s) {
        char[] chars = s.toCharArray();
        isPld = new boolean[chars.length][chars.length];
        for(int len = 1; len <= chars.length; len++) {
            for(int i = 0; i + len - 1 < chars.length; i++) {
                if(len == 1) {
                    isPld[i][i] = true;
                }else {
                    if(chars[i] == chars[i+len-1]) isPld[i][i+len-1] = (len == 2 || isPld[i+1][i+len-2]);
                }
            }
        }
        return minCut(chars,0,chars.length-1);
    }
    private int minCut(char[] chars, int start, int end) {
        if(start == end || isPld[start][end]) return 0;
        int j = end-1; 
        int min = Integer.MAX_VALUE;       
        while(j >= start) {
            if(isPld[start][j]) {
                min = Math.min(min, 1 + minCut(chars,j+1,end));
            }
            j--;
        }
        return min;
    }
}
```

但是超出了时间限制，因为递归minCut函数中必然会有重复计算。就想到用一个数组保存从位置i到位置j的最小分割次数。

```java
class Solution {
    boolean[][] isPld;
    int[][] dp;
    public int minCut(String s) {
        char[] chars = s.toCharArray();        
        isPld = new boolean[chars.length][chars.length];
        dp = new int[chars.length][chars.length];
        for(int len = 1; len <= chars.length; len++) {
            for(int i = 0; i + len - 1 < chars.length; i++) {
                if(len == 1) {
                    isPld[i][i] = true;
                }else {
                    if(chars[i] == chars[i+len-1]) isPld[i][i+len-1] = (len == 2 || isPld[i+1][i+len-2]);
                    dp[i][i+len-1] = isPld[i][i+len-1] ? 0 : -1;
                }
            }
        }
        return minCut(chars,0,chars.length-1);
    }
    private int minCut(char[] chars, int start, int end) {
        if(start == end || isPld[start][end]) return 0;
        int j = end-1; 
        int min = Integer.MAX_VALUE;       
        while(j >= start) {
            if(isPld[start][j]) {
                if(dp[j+1][end] == -1) dp[j+1][end] = minCut(chars,j+1,end);
                min = Math.min(min, 1 + dp[j+1][end]);
            }
            j--;
        }
        dp[start][end] = min;
        return min;
    }
}
```

但记录之前的计算结果这一点就已经用到了动态规划的思想了，考虑改成单纯的动态规划解法。

#### 解法二：动态规划

上面的解法中的思想如图所示：

![](https://windliang.oss-cn-beijing.aliyuncs.com/132_3.jpg)

如下图，先判断 `start` 到 `i` 是否是回文串，如果是的话，就用 `1 + d` 和之前的 `min` 比较。

![](https://windliang.oss-cn-beijing.aliyuncs.com/132_4.jpg)

然后，`i` 后移，继续判断 `start` 到 `i` 是否是回文串，如果是的话，就用 `1 + c` 和之前的 `min` 比较。

![](https://windliang.oss-cn-beijing.aliyuncs.com/132_5.jpg)

递归是栈式操作，写动态规划时要与上面相反，从左往右求每个长度的最小分割数。

用`dp[i]`记录从0到i位置这一段需要的最小切割次数。`i`从小到大遍历地求`dp[i]`，求[0,i]区间时，要遍历[0,i],[1,i],[2,i]...[i,i]。假设 `s[j,i]` 是回文串，那么 `dp[i] = Min(min,dp[j - 1])`，然后考虑所有的 `j`，也就是 `j = i, j = i - 1, j = i - 2, j = i - 3....` ，其中 `j < i` ，找出最小的。

```java
class Solution {    
    public int minCut(String s) {
        char[] chars = s.toCharArray();        
        boolean[][] isPld = new boolean[chars.length][chars.length];
        int[] dp = new int[chars.length];
        for(int len = 1; len <= chars.length; len++) {
            for(int i = 0; i + len - 1 < chars.length; i++) {
                if(len == 1) {
                    isPld[i][i] = true;
                }else {
                    if(chars[i] == chars[i+len-1]) isPld[i][i+len-1] = (len == 2 || isPld[i+1][i+len-2]);
                }
            }
        }
        for(int i = 0; i < chars.length; i++) {
            int min = Integer.MAX_VALUE;
            for(int j = 0; j <= i; j++) {
                if(isPld[j][i]) {
                    if(j == 0) {
                        min = 0;
                        break;
                    }
                    min = Math.min(min,dp[j-1]+1);
                }
            }
            dp[i] = min;
        }
        return dp[chars.length-1];
    }
}
```

这里难点就在于按照什么遍历顺序，如何去求动态规划的数组。

#### 解法三：中心扩散+动态规划

来源于[这里](https://leetcode.wang/leetcode-132-Palindrome-PartitioningII.html)。遍历每个字符，以当前字符为中心向两边扩展，判断扩展出来的是否回文串。

```
0 1 2 3 4 5 6
c f d a d f e
      ^
      c
现在以 a 为中心向两边扩展，此时第 2 个和第 4 个字符相等，我们就可以更新
dp[4] = Min(dp[4],dp[1] + 1);
也就是在当前回文串前边切一刀

然后以 a 为中心继续向两边扩展，此时第 1 个和第 5 个字符相等，我们就可以更新
dp[5] = Min(dp[5],dp[0] + 1);
也就是在当前回文串前边切一刀

然后继续扩展，直到当前不再是回文串，把中心往后移动，考虑以 d 为中心，继续更新 dp
```

```java
class Solution {
    //扩展
    public int minCut(String s) {
        int n=s.length();
        int[] dp =new int[n];
        Arrays.fill(dp,n-1);
        for(int i = 0; i < n; i++){
            // 注意偶数长度与奇数长度回文串的特点
            mincutHelper(s , i , i , dp);  // 奇数回文串以1个字符为中心
            mincutHelper(s, i , i+1 , dp); // 偶数回文串以2个字符为中心
        }
        return dp[n-1];
    }
    public void mincutHelper(String s,int i,int j,int[] dp){
        int len=s.length();
        while(i>=0&&j<len&&s.charAt(i)==s.charAt(j)){
            dp[j]=Math.min(dp[j],i==0?0:dp[i-1]+1);
            j++;
            i--;
        }
    }
}
```

以中心向外扩散的想法很妙，值得学习。

### 134. 加油站

在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

说明: 

如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。
**示例 1:**

```
输入: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

输出: 3

解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。
```

**示例 2:**

```
输入: 
gas  = [2,3,4]
cost = [3,4,3]

输出: -1

解释:
你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
因此，无论怎样，你都不可能绕环路行驶一周。
```

链接：https://leetcode-cn.com/problems/gas-station

#### 解法：贪心

首先考虑一下这道题的本质。以示例一数据为例，看一下每一个站点的gas与cost之差。

```
gas  = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]
minus= [-2,-2,-2,3,3]
```

之所以只能从站点3出发而不能从站点4出发的原因是，站点4出发后到达站点0、1会耗尽所有的油。而只有从站点3出发才能最先获得最多的油。

那么如何找到这个点？从站点`x`开始遍历，往后计算minus的和，如果从`x`到`y`的minus和值小于0，就代表从这之间任何一个点出发都不行（证明如下）。
$$
\sum_{i=x}^{y}(gas[i]-cost[i]) < 0
$$
如果有一个站点z在xy之间满足$\sum_{i=z}^{y}(gas[i]-cost[i]) > 0$，又因为$\sum_{i=x}^{y} =  \sum_{i=x}^{z-1} +\sum_{i=z}^{y}$，所以得到了$\sum_{i=x}^{z-1}(gas[i]-cost[i])<0$，所以在y之前就一定会总和小于0，与给出的条件相违背。因此代表这之间的站点都不能作为出发点。

因此就要更新出发点到y+1。然后再求从y+1到后面位置的minus和值。

如果不能绕行一周，gas总和一定小于cost总和。只要gas总和大于cost就一定有办法绕行一周。

```java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int sum = 0;
        int start = 0;
        int present = 0;
        for(int i = 0; i < gas.length; i++) {
            sum += gas[i] - cost[i];
            present += gas[i] - cost[i];
            if(present < 0) {
                start = i + 1;
                present = 0;
            }
        }
        if(sum < 0) return -1;
        return start;
    }
}
```

只用遍历一次就可以找到出发站点，写法很妙，主要是抓住了这道题的数学本质。总结来说，贪心写法的题目都是很注重数学思想。

### 135. 分发糖果

老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
评分更高的孩子必须比他两侧的邻位孩子获得更多的糖果。
那么这样下来，老师至少需要准备多少颗糖果呢？

**示例 1：**

```
输入：[1,0,2]
输出：5
解释：你可以分别给这三个孩子分发 2、1、2 颗糖果。
```

**示例 2：**

```
输入：[1,2,2]
输出：4
解释：你可以分别给这三个孩子分发 1、2、1 颗糖果。
     第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
```

链接：https://leetcode-cn.com/problems/candy

#### 解法：贪心

这题一定要确定一边后在确定另一边，先比较保证每个孩子与它左边的相对关系是正确的，在保证与他右边的相对关系是正确的。不能同时考虑！

先从前往后遍历，将每个孩子与左边的孩子比较，如果表现比左边的孩子好就要分配到比左边+1的糖果。再从后往前遍历，将每个孩子与右边的孩子进行比较，如果比右边的表现好，就要分配到比右边多的糖果。

```java
class Solution {
    public int candy(int[] ratings) {
        int l = ratings.length;
        if(l < 2) return l;
        int candy[] = new int[l];
        int num = l;
        for(int i = 1; i < l; i++) {
            if(ratings[i - 1] < ratings[i]) {
                candy[i] = candy[i - 1] + 1;
            }
        }
        for(int j = l - 2; j >= 0; j--) {
            if(ratings[j + 1] < ratings[j]) {
                candy[j] = Math.max(candy[j + 1] + 1, candy[j]);
            }
            num += candy[j + 1];
        }
        return (num + candy[0]);
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

### 136. 只出现一次的数字

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

**示例 1:**

```
输入: [2,2,1]
输出: 1
```

**示例 2:**

```
输入: [4,1,2,1,2]
输出: 4
```

链接：https://leetcode-cn.com/problems/single-number

#### 解法一：哈希表

要求线性时间复杂度，那么就不能用排序算法了。

很容易想到用哈希表来操作，用hashset。

```java
class Solution {
    public int singleNumber(int[] nums) {
        if(nums.length == 1) return nums[0];  
        HashSet<Integer> hs = new HashSet<>();      
        for(int i = 0; i < nums.length; i++) {
            if(hs.contains(nums[i])) {
                //没出现过就添加进去
                hs.remove(nums[i]);
            }else {
                //如果再次出现就删除这个元素。
                hs.add(nums[i]);
            }
        }
        return hs.iterator().next();
    }
}
```

#### 解法二：位运算

但是上面的方法使用了额外的空间，如何才能不使用额外空间？常规的方法都行不通。看到正确的做法是使用异或。

异或位运算的原理就是，相同为0，不同为1。

```
  a ⊕ b ⊕ a ⊕ b ⊕ c ⊕ c ⊕ d
= ( a ⊕ a ) ⊕ ( b ⊕ b ) ⊕ ( c ⊕ c ) ⊕ d
= 0 ⊕ 0 ⊕ 0 ⊕ d
= d
```

可以写出下面的代码。

```java
class Solution {
    public int singleNumber(int[] nums) {
        if(nums.length == 1) return nums[0];        
        for(int i = 1; i < nums.length; i++) {
            nums[i] ^= nums[i-1];
        }
        return nums[nums.length - 1];
    }
}
```

### 137. 只出现一次的数字 II

给你一个整数数组 nums ，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次 。请你找出并返回那个只出现了一次的元素。

**示例 1：**

```
输入：nums = [2,2,3,2]
输出：3
```

**示例 2：**

```
输入：nums = [0,1,0,1,0,1,99]
输出：99
```

链接：https://leetcode-cn.com/problems/single-number-ii

#### 解法一：哈希表

用一个哈希表存储数字与其出现的频次

```java
class Solution {
    public int singleNumber(int[] nums) {
        Map<Integer, Integer> freq = new HashMap<Integer, Integer>();
        for (int num : nums) {
            freq.put(num, freq.getOrDefault(num, 0) + 1);
        }
        int ans = 0;
        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
            int num = entry.getKey(), occ = entry.getValue();
            if (occ == 1) {
                ans = num;
                break;
            }
        }
        return ans;
    }
}
```

时间与空间复杂度都是线性的。

#### 解法二：状态机

[看题解](https://leetcode-cn.com/problems/single-number-ii/solution/single-number-ii-mo-ni-san-jin-zhi-fa-by-jin407891/)

![](https://pic.leetcode-cn.com/28f2379be5beccb877c8f1586d8673a256594e0fc45422b03773b8d4c8418825-Picture1.png)

对于某个二进制位，1出现的次数对三取余可以用三个状态来表示：取余得0、取余得1、取余得2。用状态机表示这三个状态之间的相互转换。

![](https://pic.leetcode-cn.com/ab00d4d1ad961a3cd4fc1840e34866992571162096000325e7ce10ff75fda770-Picture2.png)

用两个二进制位来表示 33 个状态。设此两位分别为$ two , one$则状态转换变为

![](https://pic.leetcode-cn.com/0a7ea5bca055b095673620d8bb4c98ef6c610a22f999294ed11ae35d43621e93-Picture3.png)

求two和one的表达式：

![](https://pic.leetcode-cn.com/f75d89219ad93c69757b187c64784b4c7a57dce7911884fe82f14073d654d32f-Picture4.png)

![](https://pic.leetcode-cn.com/6ba76dba1ac98ee2bb982e011fdffd1df9a6963f157b2780461dbce453f0ded3-Picture5.png)

上面的$n$是数字中的某一位，对于每一位都是同样的计算规则，所以可将以上公式直接套用在 32 位数上。有下面的代码：

```java
class Solution {
    public int singleNumber(int[] nums) {
        int ones = 0, twos = 0;
        for(int num : nums){
            ones = ones ^ num & ~twos;
            twos = twos ^ num & ~ones; // 状态机
        }
        return ones;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

#### 解法三：数电思想

基本的思想还是和上面一样的，只不过在计算表达式上，用的是真值表。

| $(a_i,b_i)$ | $x_i$ | 新$(a_i,b_i)$ |
| :---------: | :---: | :-----------: |
|     00      |   0   |      00       |
|     01      |   0   |      01       |
|     10      |   0   |      10       |
|     00      |   1   |      01       |
|     01      |   1   |      10       |
|     10      |   1   |      00       |

可以得到$a_i = a_{i}^{'}b_{i}x_{i} + a_{i}b_{i}^{'}x_{i}^{'}$，$b_{i}^{'}=a_{i}^{'}b_{i}x_{i}+a_{i}^{'}b_{i}^{'}x_{i}$。

```java
class Solution {
    public int singleNumber(int[] nums) {
        int a = 0, b = 0;
        for (int num : nums) {
            int aNext = (~a & b & num) | (a & ~b & ~num), bNext = ~a & (b ^ num);
            a = aNext;
            b = bNext;
        }
        return b;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 139. 单词拆分

给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

说明：

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。
**示例 1：**

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
```

**示例 2：**

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
```

**示例 3：**

```
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

链接：https://leetcode-cn.com/problems/word-break

#### 解法一：回溯

这题看了第一反应是用回溯法，但是提交超时了。

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        return backTracing(s, wordDict, 0);
    }
    private boolean backTracing(String s, List<String> wordDict, int start) {
        if(start == s.length()) return true;
        for(int i = start; i < s.length(); i++) {
            if(wordDict.contains(s.substring(start,i+1))) {
                if(backTracing(s, wordDict, i+1)) return true;
            }
        }
        return false;
    }
}
```

看一下超时的用例：

```
"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]
```

这样会一直反复递归到最深处，然后返回`false`。

山穷水尽了，我是搞不定这题了。

#### 解法二：动态规划

这种用递归超时无法解决的题目，似乎最终归宿都是动态规划。

`dp[j]`表示到位置`j`是否满足被空格拆分为在字典中出现的单词。判断从位置`i`到位置`j`的子字符串是否在wordDict中，如果存在，且`dp[i-1] = true`那么`dp[j] = true`。

从两种遍历方式，都可以学习一下。

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        int len = s.length();
        boolean[] dp = new boolean[len];
        for(int i = 0; i < len; i++) {
            if(i > 0 && !dp[i-1]) continue;
            for(int j = i; j < len; j++) {
                if(wordDict.contains(s.substring(i,j+1))) dp[j] = true;
            }
            if(dp[len-1]) return true;
        }
        return false;
    }
}
```



```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        int len = s.length();
        boolean[] dp = new boolean[len];
        for(int i = 0; i < len; i++) {
            for(int j = 0; j <= i; j++) {
                if((j == 0 || dp[j-1]) && wordDict.contains(s.substring(j,i+1))) {
                    //如果是从0位置开始切割或者是切割前位置处的dp值是真就进行判断
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[len-1];
    }
}
```

时间复杂度：O（n^2^）

空间复杂度：O（n）

### 140. 单词拆分 II

给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。

说明：

分隔时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。
**示例 1：**

```
输入:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
输出:
[
  "cats and dog",
  "cat sand dog"
]
```

**示例 2：**

```
输入:
s = "pineapplepenapple"
wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
输出:
[
  "pine apple pen apple",
  "pineapple pen apple",
  "pine applepen apple"
]
解释: 注意你可以重复使用字典中的单词。
```

**示例 3：**

```
输入:
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
输出:
[]
```

链接：https://leetcode-cn.com/problems/word-break-ii

#### 解法：动态规划+回溯

动态规划的思想沿用上一题。拼接字符串的部分用到了回溯算法。

```java
class Solution {
    List<String> result = new ArrayList<String>();
    StringBuilder path = new StringBuilder();
    public List<String> wordBreak(String s, List<String> wordDict) {
        int len = s.length();
        boolean [][] dp = new boolean[len][len];
        boolean[] dp1 = new boolean[len];
        for(int i = 0; i < len; i++) {
            for(int j = i; j < len; j++) {
                if(wordDict.contains(s.substring(i,j+1))) {
                    dp[i][j] = true;
                    if(!dp1[j] && ((i > 0 && dp1[i-1]) || (i == 0))) dp1[j] = true;
                }
            }
        }
        if(dp1[len-1]) backTracing(s, 0, dp);        
        return result;
    }
    private void backTracing(String s, int start, boolean[][] dp) {
        if(start == s.length()) {
            result.add(path.toString());
            return;
        }
        if(start > 0) path.append(" ");
        for(int i = start; i < s.length(); i++) {
            if(dp[start][i]) {
                path.append(s.substring(start, i+1));
                backTracing(s, i+1, dp);
                if(i != s.length() - 1) path.deleteCharAt(path.length()-1);
                path.delete(path.length()-(i+1-start),path.length());
            }
        }
    }
}
```

我用了两个数组，`dp[i][j]`表示位置`i`到位置`j`是否在wordDict当中，`dp1[i]`表示从0到i是否可以被划分。

也可以只用`dp1`这一个数组，在回溯算法里直接判断wordDict中是否包含某一段的单词即可。

### 141. 环形链表

给定一个链表，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 true 。 否则，返回 false 。

 **进阶：**

你能用 O(1)（即，常量）内存解决此问题吗？ 

**示例 1：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```

**示例 2：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test2.png)

```
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。
```

**示例 3：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test3.png)

```
输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```

链接：https://leetcode-cn.com/problems/linked-list-cycle

#### 解法：快慢指针

快慢指针是经典解法了，就相当于两个人在跑同一个圈，跑得快的那个总会和跑得慢的相遇。只要快指针和慢指针相遇，就说明存在环。

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        if(head == null) return false;
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if(fast == slow) return true;
        }  
        return false;     
    }
}
```

### 142. 环形链表 II

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

说明：不允许修改给定的链表。

进阶：

你是否可以使用 O(1) 空间解决此题？

**示例 1：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

```
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

**示例 2：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test2.png)

```
输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。
```

**示例 3：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test3.png)

```
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
```

链接：https://leetcode-cn.com/problems/linked-list-cycle-ii

#### 解法：快慢指针

设一个链表，`pos = a + 1`，参与环形链表的有`b`个节点，所以整条链表一共会有`a+b`个节点。当慢指针移动`k`步时，快指针移动了`2k`步，如果此时两个指针相遇，必定是在环上，且是慢指针第一次进入环，快指针第二次进入环。设相遇的节点是环上的第`c`个节点，所以：
$$
k=a+c
$$

$$
2k=a+b+c
$$

两式相减得到$k = b$，即慢指针已经走了`b`步，再走`a`步就可以回到`pos`位置。此时将fast指针移动回head，改变步长为`1`，当快慢指针再次相遇的时候，就是要返回的节点。

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        do{
            if( fast != null && fast.next != null ) {
                fast = fast.next.next;
                slow = slow.next;
            }else{
                return null;
            }
        }while(fast != slow);

        fast = head;
        while(fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        
        return fast;
    }
}
```

### 143. 重排链表

给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

**示例 1:**

```
给定链表 1->2->3->4, 重新排列为 1->4->2->3.
```

**示例 2:**

```
给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.
```

链接：https://leetcode-cn.com/problems/reorder-list

#### 解法一：递归

保存一个全局变量指针cur，指向需要更改next的节点。递归函数返回的节点是需要被连接到cur.next的节点。

直到cur移动到了不需要再修改连接的节点。

```java
class Solution {
    ListNode cur;
    public void reorderList(ListNode head) {
        cur = head;
        reorderListHelper(head);
    }
    private ListNode reorderListHelper(ListNode head) {
        if(head.next == null) return head;
        ListNode next = reorderListHelper(head.next);
        if(cur != next && cur.next != next) {
            //判断是否需要更改cur.next
            ListNode tmp = cur.next;
            cur.next = next;
            next.next = tmp;
            cur = tmp;
            head.next = null;
            return head;
        }else {
            return next;
        }        
    }
}
```

#### 解法二：存储节点

把链表的节点按照顺序存储在一个数组当中，然后取出对应序号的节点进行拼接即可。

```java
class Solution {
    public void reorderList(ListNode head) {
        List<ListNode> array = new ArrayList<ListNode>();
        ListNode cur = head;
        while(cur != null) {
            array.add(cur);
            cur = cur.next;
        }
        for(int i = 0; i < array.size()/2; i++) {
            cur = array.get(i);
            ListNode next = array.get(array.size()-i-1);
            next.next = cur.next;
            cur.next = next;
        }
        array.get(array.size()/2).next = null;
    }
}
```

#### 解法三：

看到一种挺妙的写法，实现过程也比较简单。

将链表从中间一分为二，将第二段链表反转后，依顺序连接两个链表。

```
1 -> 2 -> 3 -> 4 -> 5 -> 6
第一步，将链表平均分成两半
1 -> 2 -> 3
4 -> 5 -> 6

第二步，将第二个链表逆序
1 -> 2 -> 3
6 -> 5 -> 4

第三步，依次连接两个链表
1 -> 6 -> 2 -> 5 -> 3 -> 4
```

```java
class Solution {
    public void reorderList(ListNode head) {
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode newHead = slow.next;
        slow.next = null;//分割成两个链表
        newHead = reverseList(newHead);//翻转第二个链表
        ListNode cur = head;
        //连接两个链表
        while(newHead != null) {
            ListNode tmp1 = cur.next;
            ListNode tmp2 = newHead.next;
            cur.next = newHead;
            newHead.next = tmp1;
            cur = tmp1;
            newHead = tmp2;
        }
    }
    
    private ListNode reverseList(ListNode head) {
        //链表反转
        if(head == null) return head;  
        ListNode prev = null, cur = head;      
        while(cur != null) {
            ListNode tmp = cur;
            cur = cur.next;
            tmp.next = prev;
            prev = tmp;
        }
        return prev;
    }
}
```

### 144. 二叉树的前序遍历

给你二叉树的根节点 `root` ，返回它节点值的 **前序** 遍历。 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```
输入：root = [1,null,2,3]
输出：[1,2,3]
```

#### 解法一：递归

```java
class Solution {
    List<Integer> result = new ArrayList<>();
    public List<Integer> preorderTraversal(TreeNode root) {
        dfs(root);
        return result;
    }
    private void dfs(TreeNode nowNode) {
        if(nowNode != null) {
            result.add(nowNode.val);
            dfs(nowNode.left);
            dfs(nowNode.right);
        }        
    }
}
```

#### 解法二：迭代+栈

用栈模拟递归的过程。

```java
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        if (root == null) {
            return res;
        }
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        TreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                res.add(node.val);
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            node = node.right;
        }
        return res;
    }
}
```

### 145. 后序遍历

给定一个二叉树，返回它的 **后序** 遍历。

**示例:**

```
输入: [1,null,2,3]  
   1
    \
     2
    /
   3 

输出: [3,2,1]
```

进阶: 递归算法很简单，你可以通过迭代算法完成吗？
链接：https://leetcode-cn.com/problems/binary-tree-postorder-traversal

#### 解法一：递归

```java
class Solution {
    List<Integer> result = new ArrayList<>();
    public List<Integer> postorderTraversal(TreeNode root) {
        dfs(root);
        return result;
    }
    private void dfs(TreeNode nowNode) {
        if(nowNode != null) {
            dfs(nowNode.left);
            dfs(nowNode.right);
            result.add(nowNode.val);
        }
    }
}
```

#### 解法二：迭代+栈

```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        if (root == null) {
            return res;
        }
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        TreeNode prev = null;
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (root.right == null || root.right == prev) {
                //null是代表没有右节点，而root.right == prev是代表有右节点但是已经在回退
                res.add(root.val);
                prev = root;
                root = null;
            } else {
                stack.push(root);
                root = root.right;
            }
        }
        return res;
    }
}
```

后序遍历迭代的写法比较难，值得好好学习。尤其是因为要判断是否在回退。

### 150. 逆波兰表达式求值

根据[ 逆波兰表示法](https://baike.baidu.com/item/逆波兰式/128437)，求表达式的值。

有效的算符包括 `+`、`-`、`*`、`/` 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

**说明：**

- 整数除法只保留整数部分。
- 给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。

**示例 1：**

```
输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
```

**示例 2：**

```
输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
```

**示例 3：**

```
输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：
该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

链接：https://leetcode-cn.com/problems/evaluate-reverse-polish-notation

#### 解法：栈

这题是栈的经典应用题。当遇到运算符就弹栈两个操作数即可。

```java
class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<Integer>();

        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i];

            if (token.compareTo("+") == 0) {
                stack.push(stack.pop() + stack.pop());
            } else if (token.compareTo("-") == 0) {
                int nums2 = stack.pop();
                int nums1 = stack.pop();
                stack.push(nums1 - nums2);
            } else if (token.compareTo("*") == 0) {
                stack.push(stack.pop() * stack.pop());
            } else if (token.compareTo("/") == 0) {
                int nums2 = stack.pop();
                int nums1 = stack.pop();
                stack.push(nums1 / nums2);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }
}
```

也可以用数组来模拟实现栈，操作花费的时间相对于真实栈会小很多。弹栈与压栈主要是指针位置的变动。

```java
class Solution {
    public int evalRPN(String[] tokens) {
        int n = tokens.length;
        int[] stack = new int[(n + 1) / 2];
        int index = -1;
        for (int i = 0; i < n; i++) {
            String token = tokens[i];
            switch (token) {
                case "+":
                    index--;
                    stack[index] += stack[index + 1];
                    break;
                case "-":
                    index--;
                    stack[index] -= stack[index + 1];
                    break;
                case "*":
                    index--;
                    stack[index] *= stack[index + 1];
                    break;
                case "/":
                    index--;
                    stack[index] /= stack[index + 1];
                    break;
                default:
                    index++;
                    stack[index] = Integer.parseInt(token);
            }
        }
        return stack[index];
    }
}
```

### 151. 翻转字符串里的单词

给你一个字符串 `s` ，逐个翻转字符串中的所有 **单词** 。

**单词** 是由非空格字符组成的字符串。`s` 中使用至少一个空格将字符串中的 **单词** 分隔开。

请你返回一个翻转 `s` 中单词顺序并用单个空格相连的字符串。

**说明：**

- 输入字符串 `s` 可以在前面、后面或者单词间包含多余的空格。
- 翻转后单词间应当仅用一个空格分隔。
- 翻转后的字符串中不应包含额外的空格。

**示例 1：**

```
输入：s = "the sky is blue"
输出："blue is sky the"
```

**示例 2：**

```
输入：s = "  hello world  "
输出："world hello"
解释：输入字符串可以在前面或者后面包含多余的空格，但是翻转后的字符不能包括。
```

**示例 3：**

```
输入：s = "a good   example"
输出："example good a"
解释：如果两个单词间有多余的空格，将翻转后单词间的空格减少到只含一个。
```

**示例 4：**

```
输入：s = "  Bob    Loves  Alice   "
输出："Alice Loves Bob"
```

**示例 5：**

```
输入：s = "Alice does not even like bob"
输出："bob like even not does Alice"
```

链接：https://leetcode-cn.com/problems/reverse-words-in-a-string

#### 解法一：

找到每个单词的位置区间，然后再添加至StringBuilder对象。

```java
class Solution {
    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        int start = -1, end = -1;
        char[] chars = s.toCharArray();
        for(int i = 0; i < chars.length; i++) {
            if(chars[i] != ' ') {
                if(start == -1) {
                    start = i;
                }
                end = i;
            }
            if(i == chars.length - 1 || chars[i] == ' ') {
                if(start != -1) {
                    if(sb.length() != 0) sb.insert(0, " ");
                    sb.insert(0, s.substring(start, end+1));
                    start = -1;
                }
            }
        }
        return sb.toString();
    }
}
```

#### 解法二：

通过使用库函数来解决

```java
class Solution {
    public String reverseWords(String s) {
        String[] words = s.trim().split(" +");//+正则表达式
        Collections.reverse(Arrays.asList(words));
        return String.join(" ", words);
    }
}
```

#### 解法三：

题目的进阶要求是：使用 `O(1)` 额外空间复杂度的原地解法。但是java中不可以对String对象进行修改，所以只能将String转换为char数组，对char数组进行原地修改。

先翻转每个单词，再翻转整个数组，最后清除多余的空格。

```java
class Solution {
    public String reverseWords(String s) {
        char[] chars = s.toCharArray();
        //翻转每个单词
        reverseSingleWords(chars);
        //翻转整个数组
        reverseArray(chars);
        //去除多余空格
        return cleanSpace(chars);
    }
    private void reverseSingleWords(char[] chars) {   
        int start = -1, end = -1;     
        for(int i = 0; i < chars.length; i++) {
            if(chars[i] != ' ') {
                if(start == -1) {
                    start = i;
                }
                end = i;
            }
            if(i == chars.length - 1 || chars[i] == ' ') {
                if(start != -1) {
                    while(start < end) {
                        char tmp = chars[start];
                        chars[start++] = chars[end];
                        chars[end--] = tmp;
                    }
                    start = -1;
                }
            }
        }
    }
    private void reverseArray(char[] chars) {
        int start = 0, end = chars.length - 1;
        while(start < end) {
            char tmp = chars[start];
            chars[start++] = chars[end];
            chars[end--] = tmp;
        }
    }
    private String cleanSpace(char[] chars) {
        int index = 0;
        for(int i = 0; i < chars.length; i++) {
            if(chars[i] != ' ') {
                chars[index++] = chars[i];
            }else {
                //如果还没有经历过单词 或 已经有一个空格 就不需要再添加空格
                if(index == 0 || chars[index-1] == ' ') continue; 
                chars[index++] = ' ';
            }
        }
        if(chars[index-1] == ' ') index--; //可能结尾是个空格
        return new String(chars).substring(0, index);
    }
}
```

### 153. 寻找旋转排序数组中的最小值

已知一个长度为 `n` 的数组，预先按照升序排列，经由 `1` 到 `n` 次 **旋转** 后，得到输入数组。例如，原数组 `nums = [0,1,2,4,5,6,7]` 在变化后可能得到：

- 若旋转 `4` 次，则可以得到 `[4,5,6,7,0,1,2]`

- 若旋转 `7` 次，则可以得到 `[0,1,2,4,5,6,7]`

注意，数组 `[a[0], a[1], a[2], ..., a[n-1]]` **旋转一次** 的结果为数组 `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]` 。

给你一个元素值 **互不相同** 的数组 `nums` ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 **最小元素** 。

**示例 1：**

```
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
```

**示例 2：**

```
输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7] ，旋转 4 次得到输入数组。
```

**示例 3：**

```
输入：nums = [11,13,15,17]
输出：11
解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。
```

链接：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array

#### 解法：二分法

旋转数组的题也写过几道了，整体的思路都是一样的。

```java
class Solution {
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while(left < right) {
            //说明left到right区间是一个是有序递增数组
            if(nums[left] < nums[right]) break;
            //nums[left] > nums[right]
            int mid = left + (right - left)/2;
            if(nums[mid] > nums[right]) {
                left = mid+1;
            }else {
                right = mid;
            }
        }
        return nums[left];
    }
}
```

### 154. 寻找旋转排序数组中的最小值 II

已知一个长度为 `n` 的数组，预先按照升序排列，经由 `1` 到 `n` 次 **旋转** 后，得到输入数组。例如，原数组 `nums = [0,1,2,4,5,6,7]` 在变化后可能得到：

- 若旋转 `4` 次，则可以得到 `[4,5,6,7,0,1,2]`

- 若旋转 `7` 次，则可以得到 `[0,1,2,4,5,6,7]`

注意，数组 `[a[0], a[1], a[2], ..., a[n-1]]` **旋转一次** 的结果为数组 `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]` 。

给你一个可能存在 **重复** 元素值的数组 `nums` ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 **最小元素** 。

**示例 1：**

```
输入：nums = [1,3,5]
输出：1
```

**示例 2：**

```
输入：nums = [2,2,2,0,1]
输出：0
```

 #### 解法：二分法

比上一题多一点思考量的地方就在于可能有重复元素。在上一题代码的基础上做修改。

需要注意的地方就在于`nums[mid] = nums[right]`的时候可能是[3,3,1,3]，左指针要往右移动。也可能是[1,3,3]，右指针要往左移动。这是由`nums[left]`可能等于`nums[right]`引起的。这种情况的处理方法就是将右指针往左移动一个（也可以是移动left指针）。

```java
class Solution {
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while(left < right) {
            //说明left到right区间是一个是有序递增数组
            if(nums[left] < nums[right]) break;
            //如果两端值相等，就移动right指针
            while(nums[left] == nums[right] && left < right) right--;
            //nums[left] > nums[right]
            int mid = left + (right - left)/2;
            if(nums[mid] > nums[right]) {
                left = mid+1;
            }else {
                right = mid;
            }
        }
        return nums[left];
    }
}
```

### 155. 最小栈

设计一个支持 `push` ，`pop` ，`top` 操作，并能在常数时间内检索到最小元素的栈。

- `push(x)` —— 将元素 x 推入栈中。
- `pop()` —— 删除栈顶的元素。
- `top()` —— 获取栈顶元素。
- `getMin()` —— 检索栈中的最小元素。

**示例:**

```
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

链接：https://leetcode-cn.com/problems/min-stack

#### 解法一：利用系统栈

关键是要**检索到最小元素**，如果当前最小值出栈后，获得到的最小元素应该是第二小的。用系统栈的话可以利用两个栈，一个正常压栈出栈，一个用来保存新出现的最小值（如果新出现一个值，比当前最小值大，不用保存，因为这个值一定最先出栈）。

```java
class MinStack {

    /** initialize your data structure here. */
    private Stack<Integer> stackData;
    private Stack<Integer> stackMin;
    public MinStack() {
        this.stackData = new Stack<>();
        this.stackMin = new Stack<>();
    }
    
    public void push(int val) {        
        if(this.stackData.isEmpty()) {
            this.stackMin.push(val);
        }else {
            if(val <= this.getMin()) {
                this.stackMin.push(val);
            }
        }
        this.stackData.push(val);
    }
    
    public void pop() {
        if(this.stackData.isEmpty()) {
            throw new RuntimeException("Your stack is empty!!!");
        }
        if(this.getMin() == stackData.pop()) this.stackMin.pop();
    }
    
    public int top() {
        if(this.stackData.isEmpty()) {
            throw new RuntimeException("Your stack is empty!!!");
        }
        return this.stackData.peek();
    }
    
    public int getMin() {
        if(this.stackMin.isEmpty()) {
            throw new RuntimeException("Your stack is empty!!!");
        }
        return this.stackMin.peek();
    }
}
```

#### 解法二：数组实现栈

和上面的道理是一样的，不过是用数组来实现栈。

```java
class MinStack {

    /** initialize your data structure here. */
    private int[] stackData = new int[9999];
    private int[] stackMin = new int[9999];
    private int top = -1;
    public MinStack() {
    }
    
    public void push(int val) {  
        stackData[++top] = val;      
        if(top == 0) {            
            stackMin[top] = val;
        }else {
            if(stackMin[top-1] > val) {
                stackMin[top] = val;
            }else {
                stackMin[top] = stackMin[top-1];
            }
        }
    }
    
    public void pop() {
        top--;
    }
    
    public int top() {
        return stackData[top];
    }
    
    public int getMin() {
        return stackMin[top];
    }
}
```

#### 解法三：链表实现栈

创建一个私有类Node，含有min参数。（写法很妙，这样就不用两个栈了）

压栈时将创建的新节点作为头，弹栈时弹出头节点。

```java
class MinStack {

    /** initialize your data structure here. */
    private Node head;
	public MinStack() {
	}

	public void push(int val) {
    	if (head == null) {
        	head = new Node(val, val);
    	}else {
        	head = new Node(val, Math.min(val, head.min), head);
    	}
	}

	public void pop() {
    	head = head.next;
	}

	public int top() {
    	return head.val;
	}

	public int getMin() {
    	return head.min;
	}

	private class Node {
    	int val;
    	int min;
    	Node next;

    	public Node(int val, int min) {
        	this(val, min, null);
    	}

    	public Node(int val, int min, Node next) {
        	this.val = val;
        	this.min = min;
        	this.next = next;
    	}
	}
}
```

### 167. 两数之和 II - 输入有序数组

给定一个已按照 **升序排列** 的整数数组 `numbers` ，请你从数组中找出两个数满足相加之和等于目标数 `target` 。

函数应该以长度为 `2` 的整数数组的形式返回这两个数的下标值。`numbers` 的下标 **从 1 开始计数** ，所以答案数组应当满足 `1 <= answer[0] < answer[1] <= numbers.length` 。

你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

 **示例 1：**

```
输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
```

**示例 2：**

```
输入：numbers = [2,3,4], target = 6
输出：[1,3]
```

**示例 3：**

```
输入：numbers = [-1,0], target = -1
输出：[1,2]
```

链接：https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted

#### 解法：双指针

这题也没什么好说的，就是两个指针一个指向头一个指向尾，如果两数和大于target就尾指针向左移动，如果小就头指针向右移动。

```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int[] index = {0,0};
        int s = 0;
        int l = numbers.length;
        int e = l - 1;
        while(s < e) {
            int sum = numbers[s] + numbers[e];
            if(sum > target){
                e--;
            }else if(sum < target) {
                s++;
            }else {
                index[0] = s + 1;
                index[1] = e + 1;
                break;
            }
        }
        return index;
    }
}
```

### 168. Excel表列名称

给定一个正整数，返回它在 Excel 表中相对应的列名称。

例如，

    1 -> A
    2 -> B
    3 -> C
    ...
    26 -> Z
    27 -> AA
    28 -> AB 
    ...
**示例 1:**

```
输入: 1
输出: "A"
```

**示例 2:**

```
输入: 28
输出: "AB"
```

**示例 3:**

```
输入: 701
输出: "ZY"
```

链接：https://leetcode-cn.com/problems/excel-sheet-column-title

#### 解法一：迭代

这题的本质就是求26进制，但是又和真正的26进制不同，范围是`1-26`而不是`0-25`，换句话讲，正常的 `26` 进制本应该满 `26` 进 `1`，然后低位补 `0`，但是这里满 `26`就用 `26` 表示。所以对于每个还未转换的数，都先`-1`，然后再`mod26`。

```java
class Solution {
    public String convertToTitle(int columnNumber) {
        if(columnNumber == 0) return "";
        StringBuilder path = new StringBuilder();
        while(columnNumber > 0) {
            columnNumber--;
            char c =(char)(columnNumber%26 + 'A');
            path.insert(0,c);
            columnNumber = columnNumber/26;
        }
        return path.toString();
    }
}
```

#### 解法二：分治递归

观察上面代码可以发现，每次都是循环对`columNumber/26`作相同处理，其实可以通过递归调用自身函数来完成。

```java
class Solution {
    public String convertToTitle(int columnNumber) {
        if(columnNumber == 0) return "";
        columnNumber--;
        return convertToTitle(columnNumber/26) + (char)(columnNumber%26 + 'A');
    }
}
```

不过用String拼接字符串效率比较低。

### 188. 买卖股票的最佳时机 IV

给定一个整数数组 `prices` ，它的第 `i` 个元素 `prices[i]` 是一支给定的股票在第 `i` 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 **k** 笔交易。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例 1：**

```
输入：k = 2, prices = [2,4,1]
输出：2
解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。
```

**示例 2：**

```
输入：k = 2, prices = [3,2,6,5,0,3]
输出：7
解释：在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
     随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。
```

链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv

#### 解法：动态规划

这一题其实与123题是同样的思路，只不过123题规定了`k=2`，这里的`k`是个未知数。好在Java语言可以创建一个长度为变量的数组。

每一天都有`2*k`种状态，第`1`次买入，第`1`次卖出，第`2`次买入，第`2`次卖出……第`k`次买入，第`k`次卖出。创建二维数组`dp[i][j]`，表示第`i`天处于状态`j`可获得的最大收益。可以得到下面的代码。

```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        if(prices.length <= 1 || k == 0) return 0;
        int[][] dp = new int[prices.length][2*k];        
        for(int j = 0; j < 2*k; j++) {
            if(j%2==0) dp[0][j] = -prices[0];
        }
        for(int i = 1; i < prices.length; i++) {
            for(int j = 0; j < 2*k; j++) {
                if(j%2 == 0) {
                    dp[i][j] = Math.max(dp[i-1][j],(j>0?dp[i][j-1]:0)-prices[i]);
                }else {
                    dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]+prices[i]);
                };
            }
        }
        return dp[prices.length-1][2*k-1];
    }
}
```

实际上，并不需要用二维数组来保存。用一维数组就够了。

```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        int lenp = prices.length;
        if(lenp <= 1 || k == 0) return 0;
        int lenk = 2*k;
        int[] dp = new int[2*k];
        for(int i = 0; i < lenk; i++) {
            dp[i] =Integer.MIN_VALUE;
        }
        for(int i = 0; i < lenp; i++) {
            dp[0] = Math.max(dp[0], -prices[i]);
            for(int j = 0; j < lenk; j++) {
                dp[j] = Math.max(dp[j] , (j == 0 ? 0 : dp[j-1]) + (j%2 == 0 ? -prices[i] : prices[i]));
            }
        }
        return dp[lenk-1];
    }
}
```

### 198. 打家劫舍

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

**示例 1：**

```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

**示例 2：**

```
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

链接：https://leetcode-cn.com/problems/house-robber

#### 解法：动态规划

每一间房屋都有两种状态，偷与不偷。创建数组`dp[i][0]`表示不偷房屋`i`，`dp[i][1]`表示偷房屋`i`。想要偷房屋`i`必须在房屋`i-1`不被偷的基础上。所以可以得出下面的迭代式
$$
dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1])
$$

$$
dp[i][1]=dp[i-1][0]+nums[i]
$$

```java
class Solution{
    public int rob(int[] nums) {
        int[][] dp = new int[nums.length][2];
        dp[0][1] = nums[0];
        for(int i = 1; i < nums.length; i++) {
            dp[i][0] = Math.max(dp[i-1][0],dp[i-1][1]);
            dp[i][1] = dp[i-1][0]+nums[i];
        }
        return Math.max(dp[nums.length-1][0],dp[nums.length-1][1]);
    }
}
```

其实空间还可以进一步优化，只使用三个数。

```java
class Solution{
	public int rob(int[] nums) {
    	int pre2 = 0, pre1 = 0;
    	for (int i = 0; i < nums.length; i++) {
        	int cur = Math.max(pre2 + nums[i], pre1);
        	pre2 = pre1;
        	pre1 = cur;
    	}
    	return pre1;
	}
}
```

### 199. 二叉树的右视图

给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

**示例:**

```
输入: [1,2,3,null,5,null,4]
输出: [1, 3, 4]
解释:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
```

链接：https://leetcode-cn.com/problems/binary-tree-right-side-view

#### 解法一：层间遍历

通过层间遍历，将每一层最右边的节点值加入数组中

```java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if(root == null) return result;
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        while(!que.isEmpty()) {
            int size = que.size();
            TreeNode node = new TreeNode();
            for(int i = 0; i < size; i++) {
                node = que.poll();
                if(node.left != null) que.offer(node.left);
                if(node.right != null) que.offer(node.right);
            }
            result.add(node.val);
        }
        return result;
    }
}
```

层间遍历也是老朋友了，没什么可说的，唯一需要注意的就是在哪里加入结果更方便。

#### 解法二：深度优先遍历

首先，递归返回的条件一定是节点为null。

然后，确定递归调用的顺序，很容易想到一定是先递归右子节点，再去找左节点。

最后，想一下什么时候在哪里添加节点值到数组中？看下面的代码：

```java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if(root == null) return result;
        dfs(result, root, 0);
        return result;
    }
    private void dfs(List<Integer> result, TreeNode node, int level) {
        if(node == null) return;
        if(level == result.size()) {
            //当前到达的层数已经超过result的最大索引
            result.add(node.val);
        }
        dfs(result, node.right, level+1);
        dfs(result, node.left, level+1);
    }
}
```

### 200. 岛屿数量

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。 

**示例 1：**

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**示例 2：**

```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

链接：https://leetcode-cn.com/problems/number-of-islands

#### 解法一：深度优先遍历

创建一个标记数组`used[i][j]=true`表示该坐标格已经被遇到过。dfs函数向四周进行递归调用，如果上下左右四个方向上的格子是`'1'`并且还没有被标记过，就要更改标记。主函数遍历每一个格子，如果该格子是陆地，并且没有标记过就要调用dfs函数，并给岛屿数量+1。

```java
class Solution {
    int[] dir = {1, 0, -1, 0, 1};
    boolean[][] used;
    public int numIslands(char[][] grid) {
        int num=0;
        int l1=grid.length, l2=grid[0].length;
        used = new boolean[l1][l2];
        for(int i = 0; i < l1; i++) {
            for(int j = 0; j < l2; j++) {
                if(grid[i][j] == '1' && !used[i][j]) {
                    used[i][j] = true;
                    dfs(grid,i,j);
                    num++;
                }
            }
        }
        return num;
    }
    private void dfs(char[][] grid, int i, int j) {
        for(int m = 0; m < 4; m++) {
            int row = i + dir[m], col = j + dir[m+1];
            if(row >= 0 && row < grid.length && col >= 0 && col < grid[0].length) {
                if(grid[row][col] == '1' && !used[row][col]) {
                    used[row][col] = true;
                    dfs(grid,row,col);
                }
            }
        }
    }
}
```

整体的思路还是很好想的，属于经典的深度优先遍历题。对于空间复杂度还可以进一步的优化，不再多创建一个布尔数组，而是直接对grid数组进行修改。

```java
class Solution {
    int[] dir = {1, 0, -1, 0, 1};
    public int numIslands(char[][] grid) {
        int num=0;
        int l1=grid.length, l2=grid[0].length;
        for(int i = 0; i < l1; i++) {
            for(int j = 0; j < l2; j++) {
                if(grid[i][j] == '1') {
                    grid[i][j] = '0';
                    dfs(grid,i,j);
                    num++;
                }
            }
        }
        return num;
    }
    private void dfs(char[][] grid, int i, int j) {
        for(int m = 0; m < 4; m++) {
            int row = i + dir[m], col = j + dir[m+1];
            if(row >= 0 && row < grid.length && col >= 0 && col < grid[0].length) {
                if(grid[row][col] == '1') {
                    grid[row][col] = '0';
                    dfs(grid,row,col);
                }
            }
        }
    }
}
```

#### 解法二：并查集

[看这里](https://leetcode.wang/leetcode-200-Number-of-Islands.html)

用 `nums` 来记录当前有多少个岛屿，初始化的时候每个 `1` 都认为是一个岛屿，然后开始合并。

遍历每个为 `1` 的节点，将它的右边和下边的 `1` 和当前节点合并（不需要上下左右，遍历顺序是从左到右从上到下，不像解法一）。每进行一次合并，就将 `nums` 减 `1` 。

最后返回 `nums` 。

```java
class UnionFind {
    int[] parents;
    int nums;

    public UnionFind(char[][] grid, int rows, int cols) {
        nums = 0;
        // 记录 1 的个数
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '1') {
                    nums++;
                }
            }
        }

        //每一个类初始化为它本身
        int totalNodes = rows * cols;
        parents = new int[totalNodes];
        for (int i = 0; i < totalNodes; i++) {
            parents[i] = i;
        }
    }

    void union(int node1, int node2) {
        //注意是找到他们的根节点，对根进行合并
        int root1 = find(node1);
        int root2 = find(node2);
        //发生合并，nums--
        if (root1 != root2) {
            parents[root2] = root1;
            nums--;
        }
    }

    int find(int node) {
        while (parents[node] != node) {
            parents[node] = parents[parents[node]];
            node = parents[node];
        }
        return node;
    }

    int getNums() {
        return nums;
    }
}

int rows;
int cols;

public int numIslands(char[][] grid) {
    if (grid.length == 0)
        return 0;

    rows = grid.length;
    cols = grid[0].length;
    UnionFind uf = new UnionFind(grid, rows, cols);

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            if (grid[row][col] == '1') {
                // 将下边右边的 1 节点和当前节点合并
                if (row != (rows - 1) && grid[row + 1][col] == '1') {
                    uf.union(node(row, col), node(row + 1, col));
                }
                if (col != (cols - 1) && grid[row][col + 1] == '1') {
                    uf.union(node(row, col), node(row, col + 1));
                }
            }
        }
    }
    return uf.getNums();

}

int node(int i, int j) {
    return i * cols + j;//计算在一维数组中的坐标
}

```

### 202. 快乐数

编写一个算法来判断一个数 `n` 是不是快乐数。

「快乐数」定义为：

- 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
- 然后重复这个过程直到这个数变为 1，也可能是 **无限循环** 但始终变不到 1。
- 如果 **可以变为** 1，那么这个数就是快乐数。

如果 `n` 是快乐数就返回 `true` ；不是，则返回 `false` 。

**示例 1：**

```
输入：19
输出：true
解释：
1 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
```

**示例 2：**

```
输入：n = 2
输出：false
```

链接：https://leetcode-cn.com/problems/happy-number

#### 解法一：哈希集合

通过试验几个数发现，最后总会陷入一个死循环，以116为例：

![](https://assets.leetcode-cn.com/solution-static/202/202_fig2.png)

所以通过将所有遇到过的数字都加入到哈希集合中，如果遇到了相同数，且这个数不为1的话，就返回`false`。

```java
class Solution {
    private int getNext(int n) {
        int totalSum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalSum += d * d;
        }
        return totalSum;
    }

    public boolean isHappy(int n) {
        Set<Integer> seen = new HashSet<>();
        while (n != 1 && !seen.contains(n)) {
            seen.add(n);
            n = getNext(n);
        }
        return n == 1;
    }
}
```

#### 解法二：快慢指针

观察上面的图能联想到环形链表查找环形入口，所以这题也可以用同样的方法来实现：设置两个指针，快指针每次执行两次getNext，慢指针执行一次getNext。

```java
class Solution {
    public boolean isHappy(int n) {
        int slow = n, fast = n;
        //注意这里不能先判断while(slow!=fast),因为刚开始slow和fast值相同
        do{
            slow = getNext(slow);
            fast = getNext(getNext(fast));
        }while(slow != fast);
        return slow == 1;
    }
    private int getNext(int n) {
        int totalSum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalSum += d * d;
        }
        return totalSum;
    }
}
```

### 203. 移除链表元素

给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/03/06/removelinked-list.jpg)

```
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]
```

**示例 2：**

```
输入：head = [], val = 1
输出：[]
```

**示例 3：**

```
输入：head = [7,7,7,7], val = 7
输出：[]
```

 连接：https://leetcode-cn.com/problems/remove-linked-list-elements/

#### 解法一：迭代

首先观察示例，发现有遇到头节点需要删除的情况，所以为了处理方便，就添加一个虚拟头结点。

```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode fakeHead = new ListNode(0,head);
        ListNode cur = fakeHead;
        //此处cur.next!=null注意不要遗漏
        while(cur != null && cur.next != null) {
            if(cur.next.val != val) {
                cur = cur.next;
            }else {
                cur.next = cur.next.next;   
            }                     
        }
        return fakeHead.next;
    }
}
```

#### 解法二：递归

```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if(head == null) return null;
        if(head.val == val) {
            return removeElements(head.next,val);
        }else {
            head.next = removeElements(head.next,val);
            return head;
        }
    }
}
```

### 204. 计数质数

统计所有小于非负整数 *`n`* 的质数的数量。

**示例 1：**

```
输入：n = 10
输出：4
解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。
```

**示例 2：**

```
输入：n = 0
输出：0
```

**示例 3：**

```
输入：n = 1
输出：0
```

链接：https://leetcode-cn.com/problems/count-primes

#### 解法一：暴力

判断一个数是否为质数：即这个数是否有除1和其自身以外的因子。

```java
class Solution {
    public int countPrimes(int n) {
        int cnt = 0;
        for(int i = 2; i < n; i++) {
            if(isPrime(i)) cnt++;
        }
        return cnt;
    }
    private boolean isPrime(int n) {
        for(int i = 2; i < n; i++) {
            if(n%i == 0) return false;
        }
        return true;
    }
}
```

时间复杂度：O（n^2^）

这种方法会超时，考虑优化。

其实并不需要判断从2到n-1所有的数，只需要判断2到根号n即可，因为如果n有一个因子是比根号n大的数，那么必定有对应的另一个因子小于根号n

```java
class Solution {
    public int countPrimes(int n) {
        int cnt = 0;
        for(int i = 2; i < n; i++) {
            if(isPrime(i)) cnt++;
        }
        return cnt;
    }
    private boolean isPrime(int n) {
        int m = (int)Math.sqrt(n);
        for(int i = 2; i <= m; i++) {
            if(n%i == 0) return false;
        }
        return true;
    }
}
```

时间复杂度：O（n$\sqrt{n}$）

但是这种方法也超时了。

#### 解法二：厄拉多塞筛法

用一个布尔数组表示当前数是否是素数。

然后从 `2` 开始，将 `2` 的倍数，`4`、`6`、`8`、`10` ...依次标记为非素数。

下个素数 `3`，将 `3` 的倍数，`6`、`9`、`12`、`15` ...依次标记为非素数。

下个素数 `5`，将 `5` 的倍数，`10`、`15`、`20`、`25` ...依次标记为非素数。

实际上执行的时候，并不需要从2 * i，3 * i开始，直接从i^2^开始即可，原因很好想这里不赘述了。

在代码中，因为数组默认值是 `false` ，所以用 `false` 代表当前数是素数，用 `true` 代表当前数是非素数。

```java
class Solution {
    public int countPrimes(int n) {
        boolean[] notPrim = new boolean[n];
        int cnt = 0;
        for(int i = 2; i < n; i++) {
            if(!notPrim[i]) {
                cnt++;
                for(long j = (long)(i)*i; j < n; j+=i) {
                    notPrim[(int)j] = true;
                }
            }
        }
        return cnt;
    }
}
```

时间复杂度：O*(*nloglogn)。

### 205. 同构字符串

给定两个字符串 s 和 t，判断它们是否是同构的。

如果 s 中的字符可以按某种映射关系替换得到 t ，那么这两个字符串是同构的。

每个出现的字符都应当映射到另一个字符，同时不改变字符的顺序。不同字符不能映射到同一个字符上，相同字符只能映射到同一个字符上，字符可以映射到自己本身。

**示例 1:**

```
输入：s = "egg", t = "add"
输出：true
```

**示例 2：**

```
输入：s = "foo", t = "bar"
输出：false
```

**示例 3：**

```
输入：s = "paper", t = "title"
输出：true
```

链接：https://leetcode-cn.com/problems/isomorphic-strings

#### 解法：哈希表

维护两张哈希表，第一张哈希表以 s 中字符为键，映射至 t 的字符为值，第二张哈希表以 t 中字符为键，映射至 s 的字符为值。从左至右遍历两个字符串的字符，不断更新两张哈希表，如果出现冲突（即当前下标index 对应的字符 s[index] 已经存在映射且不为 t[index] 或当前下标 index 对应的字符 t[index] 已经存在映射且不为s[index]）时说明两个字符串无法构成同构，返回false。

```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> s2t = new HashMap<Character, Character>();
        Map<Character, Character> t2s = new HashMap<Character, Character>();
        int len = s.length();
        for (int i = 0; i < len; ++i) {
            char x = s.charAt(i), y = t.charAt(i);
            if ((s2t.containsKey(x) && s2t.get(x) != y) || (t2s.containsKey(y) && t2s.get(y) != x)) {
                return false;
            }
            s2t.put(x, y);
            t2s.put(y, x);
        }
        return true;
    }
}
```

其实并不需要保存两个的对应关系，只要s[index]和t[index]上一次所在的位置不同就表明一定是false。（最开始想的是利用出现的个数来判断，但是后来发现当遇到`aaabbbb`和`bbbaaab`这样的测试样例时，就会误判）

进一步优化不需要用哈希表来存储，只需要用一个数组就可以了。每一个char字符本身就是一个索引建。

```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        int[] as = new int[128];
        int[] at = new int[128];
        for(int i = 0; i < s.length(); i++) {
            char ss = s.charAt(i), tt = t.charAt(i);
            if(as[ss] != at[tt]) return false;
            as[ss] = i + 1;
            at[tt] = i + 1;
        }
        return true;
    }
}
```

### 206. 反转链表

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)

```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex2.jpg)

```
输入：head = [1,2]
输出：[2,1]
```

**示例 3：**

```
输入：head = []
输出：[]
```

链接：https://leetcode-cn.com/problems/reverse-linked-list/

#### 解法一：递归

递归有两种写法，一个方法是先递归后翻转：

```java
class Solution {
    ListNode newHead = null;//全局变量保存新的头节点
    public ListNode reverseList(ListNode head) {
        if(head == null) return null;
        reverseHelper(head);
        return newHead;
    }
    private ListNode reverseHelper(ListNode head) {
        if(head.next == null) {
            newHead = head;
            return head;
        }
        ListNode last = reverseHelper(head.next);
        head.next = null;//断开当前节点与后面节点的连接
        last.next = head;//将当前节点连接到新的链表中
        return head;
    }
}
```

还有一种方法是先翻转再递归，此时就不再需要一个全局变量保存头节点了。

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null) return head;        
        return reverseList(null,head);
    }
    public ListNode reverseList(ListNode prev, ListNode cur) {
        if(cur == null) return prev;
        ListNode tmp = cur;
        cur = cur.next;
        tmp.next = prev;
        prev = tmp;
        return reverseList(prev, cur);
    }
}
```

#### 解法二：迭代

递归方法二可以改写成迭代的形式：

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null) return head;  
        ListNode prev = null, cur = head;      
        while(cur != null) {
            ListNode tmp = cur;
            cur = cur.next;
            tmp.next = prev;
            prev = tmp;
        }
        return prev;
    }
}
```

### 209. 长度最小的子数组

给定一个含有 `n` 个正整数的数组和一个正整数 `target`。

找出该数组中满足其和 `≥ target` 的长度最小的 **连续子数组** `[numsl, numsl+1, ..., numsr-1, numsr]` ，并返回其长度**。**如果不存在符合条件的子数组，返回 `0` 。

**示例 1：**

```
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
```

**示例 2：**

```
输入：target = 4, nums = [1,4,4]
输出：1
```

**示例 3：**

```
输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0
```

链接：https://leetcode-cn.com/problems/minimum-size-subarray-sum

#### 解法一：双指针

很快就想到一个双指针（也可以看作是一个长度可变的滑动窗口）的思路，左右各一个指针，在`prevSum`小于`target`时，右指针一直往右。当`prevSum`满足大于等于`target`条件，就将左指针往左移动，并更新`min`。

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        //prevSum保存left到right区间的加和
        int prevSum = 0, left = 0, right = 0;
        int min = 0;//保存最小个数
        while(right < nums.length) {
            prevSum += nums[right];
            while(prevSum >= target && left <= right) {
                if(min == 0 || min > right-left+1) {
                    min = right - left + 1;
                }
                prevSum -= nums[left++];
            }
            right++;
        }
        return min;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

题目给出了进阶的要求，让实现一个时间复杂度为O（nlogn）的算法，并且还有`二分法`的标签，没有任何思路，看题解了。

#### 解法二：二分法

新开一个数组，保存的是原数组的前缀和，`sum[i+1]`代表的是`nums[0]+nums[1]+...+nums[i]`。

因为题目说明数组中的值都是正值，所以可以得知`sum[i]`一定是递增的序列，那么就可以用二分查找的方法了。

只需要找到 `sums[k]-sums[j]>=target`，那么 `k-j` 就是满足的连续子数组，求 `sums[k]-sums[j]>=target` 可以通过求 `sums[j]+target<=sums[k]`。

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int[] sum = new int[nums.length+1];
        for(int i = 1; i <= nums.length; i++) {
            sum[i] = sum[i-1] + nums[i-1];
        }
        int min = 0;
        for(int i = 0; i <= nums.length; i++) {
            int pos = binarySearch(sum, sum[i]+target);
            if(pos <= nums.length && (min == 0 || min > pos-i)) min = pos-i;
        }
        return min;
    }
    private int binarySearch(int[] nums, int target) {
        int left = 0, right = nums.length-1;
        while(left <= right) {
            int mid = left + (right - left)/2;
            if(nums[mid] == target) return mid;
            if(nums[mid] < target) {
                left = mid+1;
            }else {
                right = mid-1;
            }
        }
        //如果没有找到这个值，返回如果要在数组中插入这个值所应该在的位置
        return left;
    }
}
```

时间复杂度：O（nlogn）

空间复杂度：O（n）

利用前缀和构造一个递增序列的思想值得学习。

### 213. 打家劫舍 II

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 **围成一圈** ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警** 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **在不触动警报装置的情况下** ，今晚能够偷窃到的最高金额。

**示例 1：**

```
输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
```

**示例 2：**

```
输入：nums = [1,2,3,1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

**示例 3：**

```
输入：nums = [0]
输出：0
```

链接：https://leetcode-cn.com/problems/house-robber-ii

#### 解法：动态规划

198题也是打家劫舍，与这题不同的地方在于，198题中的房屋是单排的，而这一题的房屋围成了一个圈，这就代表第一间房屋和最后一间房屋只能选一个偷。

将0-n的房屋分成0-n-1和1-n的两组单排房屋。取这两个结果中的最大值即可。

```java
class Solution {
    public int rob(int[] nums) {
        int len = nums.length;
        if(len == 1) return nums[0];
        if(len == 2) return Math.max(nums[0],nums[1]);
        return Math.max(rob(nums, 0, len - 2),rob(nums, 1, len - 1));
        
    }
    private int rob(int[] nums, int start, int end) {
        int pre2 = 0, pre1 = 0;
        for (int i = start; i <= end; i++) {
            int cur = Math.max(pre2 + nums[i], pre1);
            pre2 = pre1;
            pre1 = cur;
        }
        return pre1;
    } 
}
```

### 215. 数组中的第K个最大元素

在未排序的数组中找到第 **k** 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

**示例 1:**

```
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
```

**示例 2:**

```
输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
```

链接：https://leetcode-cn.com/problems/kth-largest-element-in-an-array

#### 解法一：暴力

最直接的想法就是数组排序完直接返回`nums[len-k]`，因为题目要求的是*第 k 个最大的元素，而不是第 k 个不同的元素*。

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - k];
    }
}
```

时间复杂度：O（nlogn）

空间复杂度：O（1）

因为排序算法系统默认使用快排，它的时间复杂度是O（nlogn）。

#### 解法二：partition

用的是快排中的思想，快速排序算法中每次选定一个数，然后将所有比它小的数扔到它前面去，所有比它大的数扔到他后面。

> partition（切分）操作总能排定一个元素，还能够知道这个元素它最终所在的位置，这样每经过一次 partition（切分）操作就能缩小搜索的范围，这样的思想叫做 “减而治之”（是 “分而治之” 思想的特例）。

每次随机选定一个数(如果不随机选定，最差的情况时间复杂度会达到n^2^)，得到它在数组中排序后应当在的位置`index`，将它与`len-k`比较大小，如果比`len-k`大，就对`index`左边继续切分，如果比它小，就往`index`右边。

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        return sort(nums, 0, nums.length - 1, nums.length - k);
    }
    private int sort(int[] nums, int lo, int hi, int target) {
        if(hi <= lo) return nums[lo];
        //生成随机切分点
        Random rand = new Random();
        int randInd = lo + rand.nextInt(hi - lo);
        exch(nums, lo, randInd);//将这个数移到数组的开始去

        int i = lo, j = hi + 1, v = nums[lo];
        while(true) {
            while(nums[++i] < v) {
                if(i == hi) break;
            }//i指针一直移动到一个比v大的数才停止
            while(v < nums[--j]) {
                if(j == lo) break;
            }//j指针一直移动到一个比v小的数才停止
            if(i >= j) break;
            exch(nums, i, j);//将刚刚找到的两个数交换位置
        }
        exch(nums, lo, j);//将该数移动到该在的位置
        
        if(j == target) return nums[j];
        if(j > target) {
            return sort(nums, lo, j - 1, target);
        }else {
            return sort(nums, j + 1, hi, target);
        }
    }
    private void exch(int[] nums, int i, int j) {
        int store = nums[i];
        nums[i] = nums[j];
        nums[j] = store;
    }
}
```

时间复杂度：O（n），需要用到主定理，可以看[这里](https://blog.csdn.net/qq_41739364/article/details/101224786)

空间复杂度：O（1）

#### 解法三：优先队列

有两种做法：

- 维护一个最小堆，弹出`len-k`个元素后，此时最小堆只剩下 `k` 个元素，堆顶元素就是数组中的第 `k` 个最大元素。
- 维护一个最大堆，弹出`k-1`个元素后，此时最大堆的堆顶元素就是数组中的第 `k` 个最大元素。

也可以结合上面两种写法，当`k>n/2`的时候，用最小堆，`k<n/2`的时候，用最大堆。

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int len = nums.length;
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>(len,(a,b)->a-b);
        for(int i = 0; i < len; i++) {
            pq.offer(nums[i]);
        }
        for(int i = 0; i < len - k; i++) {
            pq.poll();
        }
        return pq.poll();
    }
}
```

时间复杂度：O（nlogn）

空间复杂度：O（n）

### 217. 存在重复元素

给定一个整数数组，判断是否存在重复元素。

如果存在一值在数组中出现至少两次，函数返回 `true` 。如果数组中每个元素都不相同，则返回 `false` 。

**示例 1:**

```
输入: [1,2,3,1]
输出: true
```

**示例 2:**

```
输入: [1,2,3,4]
输出: false
```

**示例 3:**

```
输入: [1,1,1,3,3,4,3,2,4,2]
输出: true
```

链接：https://leetcode-cn.com/problems/contains-duplicate

#### 解法一：哈希表

```java
class Solution {
    public boolean containsDuplicate(int[] nums) {
        HashSet<Integer> hs = new HashSet<>();
        for(int i = 0; i < nums.length; i++) {
            hs.add(nums[i]);
        }
        return nums.length > hs.size();
    }
}
```

#### 解法二：快速排序

找重复元素，可以通过排序。这里可以自己写一下快速排序，并添加条件，当排序过程中一旦发现了有重复元素就返回。

```java
class Solution {
    public boolean containsDuplicate(int[] nums) {
        return sort(nums, 0, nums.length-1);
    }
    
    private boolean sort(int[] nums, int lo, int hi) {
        if(hi <= lo) return false;
        //生成随机切分点
        Random rand = new Random();
        int randInd = lo + rand.nextInt(hi - lo);
        exch(nums, lo, randInd);//将这个数移到待排序数组的头

        int i = lo, j = hi + 1, v = nums[lo];
        while(true) {
            while(nums[++i] < v) {
                if(i == hi) break;
            }
            if(nums[i]==v) return true;
            while(v < nums[--j]) {
                if(j == lo) break;
            }
            if(i >= j) break;
            if(nums[j]==v) return true;
            exch(nums, i, j);//将刚刚找到的两个数交换位置
        }
        exch(nums, lo, j);//将该数移动到该在的位置
        return sort(nums, lo, j - 1) || sort(nums, j + 1, hi);
        
    }
    private void exch(int[] nums, int i, int j) {
        int store = nums[i];
        nums[i] = nums[j];
        nums[j] = store;
    }
}
```

### 222. 完全二叉树的节点个数

给你一棵 **完全二叉树** 的根节点 `root` ，求出该树的节点个数。[完全二叉树](https://baike.baidu.com/item/完全二叉树/7773232?fr=aladdin) 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 `h` 层，则该层包含 `1~ 2h` 个节点。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/01/14/complete.jpg)

```
输入：root = [1,2,3,4,5,6]
输出：6
```

**示例 2：**

```
输入：root = []
输出：0
```

**示例 3：**

```
输入：root = [1]
输出：1
```

链接：https://leetcode-cn.com/problems/count-complete-tree-nodes

#### 解法一：当做普通的二叉树

求一棵树的节点有递归和迭代两种做法。

```java
class Solution {
    public int countNodes(TreeNode root) {
        if(root == null) return 0;
        return 1+countNodes(root.left)+countNodes(root.right);
    }
}
```

```java
class Solution {
    public int countNodes(TreeNode root) {
        if(root == null) return 0;
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        que.offer(root);
        int r = 0;
        while(!que.isEmpty()) {
            r++;
            TreeNode node = que.poll();
            if(node.left != null) que.offer(node.left);
            if(node.right != null) que.offer(node.right);
        }
        return r;
    }
}
```

#### 解法二：二分查找满二叉树

对于一颗完全二叉树，它的子树中一定存在满二叉树。

如果右子树的高度等于整个树的高度减 `1`，说明左边都填满了，所以左子树是 perfect binary tree ，如下图。

![img](https://windliang.oss-cn-beijing.aliyuncs.com/222_6.jpg)

否则的话，右子树是 perfect binary tree ，如下图。

![img](https://windliang.oss-cn-beijing.aliyuncs.com/222_7.jpg)

满二叉树的计算公式是$2^h-1$。

所以可以通过计算右边树的高度判断哪一边是满二叉树，利用公式计算出满二叉树的节点，再区递归计算非满二叉树（它的子树中必定有满二叉树）的节点。

```java
class Solution {
    public int countNodes(TreeNode root) {
        if(root == null) return 0;
        int height = getHeight(root);
        int rightHeight = getHeight(root.right);
        if(rightHeight < height-1) {
            //右子树是满二叉树
            return (1 << rightHeight) + countNodes(root.left);
        }else {
            //左子树是满二叉树
            return (1 << rightHeight) + countNodes(root.right);
        }
    }
    private int getHeight(TreeNode root) {
        if(root == null) return 0;
        return 1 + getHeight(root.left);
    }
}
```

时间复杂度：使用了类似二分的思想，每次都去掉了二叉树一半的节点，所以总共会进行 `O(log(n))` 次。每次求高度消耗 `O(log(n))` 。因此总的时间复杂度是 `O(log²(n))`。

### 225. 用队列实现栈

请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通队列的全部四种操作（`push`、`top`、`pop` 和 `empty`）。

实现 `MyStack` 类：

- `void push(int x)` 将元素 x 压入栈顶。
- `int pop()` 移除并返回栈顶元素。
- `int top()` 返回栈顶元素。
- `boolean empty()` 如果栈是空的，返回 `true` ；否则，返回 `false` 。

**示例：**

```
输入：
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 2, 2, false]

解释：
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // 返回 2
myStack.pop(); // 返回 2
myStack.empty(); // 返回 False
```

链接：https://leetcode-cn.com/problems/implement-stack-using-queues

#### 解法一：双端队列

双端队列既能从头出也能从尾出，没什么技术含量不写了。

#### 解法二：两个队列

当要出栈的时候，就把主队列的除最后一个元素以外的所有元素加入副队列中，清除掉所有主队列元素后，把副队列的所有元素再加入主队列中。

```java
class MyStack {
    private Queue<Integer> que1;
    private Queue<Integer> que2;
    /** Initialize your data structure here. */
    public MyStack() {
        que1 = new LinkedList<Integer>();
        que2 = new LinkedList<Integer>();
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        que1.offer(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        int size = que1.size();
        while(--size > 0) {
            que2.offer(que1.poll());
        }
        int res = que1.poll();
        while(!que2.isEmpty()) {
            que1.offer(que2.poll());
        }
        return res;
    }
    
    /** Get the top element. */
    public int top() {
        int res = 0;
        while(!que1.isEmpty()) {
            res = que1.poll();
            que2.offer(res);
        }        
        while(!que2.isEmpty()) {
            que1.offer(que2.poll());
        }
        return res;
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return que1.isEmpty() && que2.isEmpty();
    }
}
```

#### 解法三：一个队列

上面解法二用了两个队列，实际上并不需要，可以只用一个队列。当要执行`pop()`出栈的时候，把队列中`size-1`的元素弹出并重新加入到队列中，最后再弹出一个元素即可。

```java
class MyStack {
    private Queue<Integer> que;
    /** Initialize your data structure here. */
    public MyStack() {
        que = new LinkedList<Integer>();
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        que.offer(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        int size = que.size();
        while(--size > 0) {
            que.offer(que.poll());
        }
        int res = que.poll();
        return res;
    }
    
    /** Get the top element. */
    public int top() {
        int size = que.size();
        while(--size > 0) {
            que.offer(que.poll());
        }
        int res = que.poll();
        que.offer(res);
        return res;
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return que.isEmpty();
    }
}
```

### 226. 翻转二叉树

翻转一棵二叉树。

**示例：**

输入：

        4
      /   \
      2     7
     / \   / \
    1   3 6   9

输出：

        4
      /   \
      7     2
     / \   / \
    9   6 3   1

链接：https://leetcode-cn.com/problems/invert-binary-tree

#### 解法一：递归

递归可以有前序、中序、后序三种写法，相对比较简单，都列举一下：

前序：

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null) return null;
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }
}
```

中序：

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null) return null;
        invertTree(root.left);
        TreeNode tmp = root.right;
        root.right = root.left;
        root.left = tmp;
        invertTree(root.left);
        return root;
    }
}
```

后序：

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null) return null;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.right = left;
        root.left = right;
        return root;
    }
}
```

#### 解法二：迭代+栈

用栈来模拟递归的写法。

前序：

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null) return null;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()) {
            TreeNode cur = stack.pop();
            TreeNode tmp = cur.left;
            cur.left = cur.right;
            cur.right = tmp;
            if(cur.left != null) stack.push(cur.left);
            if(cur.right != null) stack.push(cur.right);
        }
        return root;
    }
}
```

中序：

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null) return null;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode res = root;
        root = root.left;
        while(root != null || !stack.isEmpty()) {
            while(root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            TreeNode tmp = root.right;
            root.right = root.left;
            root.left = tmp;
            root = root.left;
        }
        return res;
    }
}
```

后序：

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null) return null;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode res = root;
        root = root.left;
        TreeNode prev = null;
        while(root != null || !stack.isEmpty()) {
            while(root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (root.right == null || root.right == prev) {
                //null是代表没有右节点，而root.right == prev是代表有右节点但是已经在回退
                TreeNode tmp = root.left;
                root.left = root.right;
                root.right = tmp;
                prev = root;
                root = null;
            } else {
                stack.push(root);
                root = root.right;
            }
        }
        return res;
    }
}
```

其中，后序的写法比较难，值得多多推敲。

#### 解法三：层间遍历

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null) return null;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        TreeNode cur;
        queue.offer(root);
        while(!queue.isEmpty()) {
            cur = queue.poll();
            TreeNode tmp = cur.left;
            cur.left = cur.right;
            cur.right = tmp;
            if(cur.left != null) queue.offer(cur.left);
            if(cur.right != null) queue.offer(cur.right);
        }
        return root;
    }
}
```

和上面用栈做前序遍历是基本一样的写法，只不过真正实际遍历的顺序有所改变。

### 232. 用栈实现队列

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（`push`、`pop`、`peek`、`empty`）：

实现 `MyQueue` 类：

- `void push(int x)` 将元素 x 推到队列的末尾
- `int pop()` 从队列的开头移除并返回元素
- `int peek()` 返回队列开头的元素
- `boolean empty()` 如果队列为空，返回 `true` ；否则，返回 `false`

**进阶：**

- 你能否实现每个操作均摊时间复杂度为 `O(1)` 的队列？换句话说，执行 `n` 个操作的总时间复杂度为 `O(n)` ，即使其中一个操作可能花费较长时间。

 #### 解法：两个栈

和前面225题用两个队列实现栈有异曲同工之妙。

```java
class MyQueue {
    private Stack<Integer> stackPush;
    private Stack<Integer> stackPop;
    /** Initialize your data structure here. */
    public MyQueue() {
        this.stackPush = new Stack<>();
        this.stackPop = new Stack<>();
    }
    
    /** Push element x to the back of queue. */
    public void push(int x) {
        this.stackPush.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if(this.stackPop.isEmpty()) {
            if(this.stackPush.isEmpty()) return -1;
            while(!this.stackPush.isEmpty()) {
                this.stackPop.push(stackPush.pop());
            }
        }
        return this.stackPop.pop();
    }
    
    /** Get the front element. */
    public int peek() {
        if(this.stackPop.isEmpty()) {
            if(this.stackPush.isEmpty()) return -1;
            while(!this.stackPush.isEmpty()) {
                this.stackPop.push(stackPush.pop());
            }
        }
        return stackPop.peek();
    }
    
    /** Returns whether the queue is empty. */
    public boolean empty() {
        return this.stackPop.isEmpty() && this.stackPush.isEmpty();
    }
}
```

### 235. 二叉搜索树的最近公共祖先

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/binarysearchtree_improved.png) 

**示例 1:**

```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。
```

**示例 2:**

```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
```

链接：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree

#### 解法一：递归

这题有一个比较好的前提条件：二叉搜索树。

- 如果给定的两个节点的值都小于根节点的值，那么最近的共同祖先一定在左子树

- 如果给定的两个节点的值都大于根节点的值，那么最近的共同祖先一定在右子树

- 如果一个大于等于、一个小于等于根节点的值，那么当前根节点就是最近的共同祖先了

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if((root.val >= p.val && root.val <= q.val) || (root.val <= p.val && root.val >= q.val)) return root;
        if(root.val > p.val && root.val > q.val) return lowestCommonAncestor(root.left, p, q);
        //if(root.val < p.val && root.val < q.val) 
        return lowestCommonAncestor(root.right, p, q);
    }
}
```

#### 解法二：迭代

上面的递归还可以改成迭代的写法

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        int min = Math.min(p.val,q.val), max = Math.max(p.val,q.val);
        while(root.val > max || root.val < min) {
            if(root.val > max) {
                root = root.left;
            }else if(root.val < min) {
                root = root.right;
            }
        }
        return root;
    }
}
```

### 236. 二叉树的最近公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

**示例 1：**

![](https://assets.leetcode.com/uploads/2018/12/14/binarytree.png)

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2018/12/14/binarytree.png)

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
```

**示例 3：**

```
输入：root = [1,2], p = 1, q = 2
输出：1
```

链接：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree

#### 解法一：

常规思路，分别搜索从根节点到p、q的路径，然后比较路径的分叉点。

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        List<TreeNode> path1 = new ArrayList<>();
        List<TreeNode> path2 = new ArrayList<>();
        findPath(root, p, path1);
        findPath(root, q, path2);
        int i = 0;
        for(; i < Math.min(path1.size(), path2.size()); i++) {
            if(path1.get(i) != path2.get(i)) break;
        }
        return path1.get(i-1);
    }
    private void findPath(TreeNode root, TreeNode target, List<TreeNode> path) {
        path.add(root);
        if(root == target || root == null) {
            return;
        }
        int size = path.size();
        findPath(root.left, target, path);
        if(path.get(path.size() - 1) == target) return;
        for(int i = size; i < path.size(); i++) {
            path.remove(i);
        }
        findPath(root.right, target, path);
    }
}
```

其实整个算法有重复的部分，搜寻路径的时候，可能就已经包含了两个节点，但是还是得重新来一遍。

#### 解法二：

思路是用一个递归函数，如果当前节点是需要搜索的p或者q就返回，如果是null也要返回。若不满足返回的要求，就要继续向其左右子节点寻找，如果它左边和右边递归函数返回的都不是null，那么就说明，该节点本身就是公共祖先。如果左右边有一个是null，那么就取非null的那一边作为公共节点。

很是巧妙，所有节点最多只需要遍历一次就可以得到结果。

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left == null) {
            return right;
        }
        if(right == null) {
            return left;
        }
        return root;
    }
}
```

### 239. 滑动窗口

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

**示例 1：**

```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值

---------------               -----

[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**示例 2：**

```
输入：nums = [1], k = 1
输出：[1]
```

**示例 3：**

```
输入：nums = [1,-1], k = 1
输出：[1,-1]
```

**示例 4：**

```
输入：nums = [9,11], k = 2
输出：[11]
```

**示例 5：**

```
输入：nums = [4,-2], k = 2
输出：[4]
```

链接：https://leetcode-cn.com/problems/sliding-window-maximum

#### 解法一：双头队列单调栈

利用双头队列构造一个单调递减的队列。如果新的值大于队尾的值，那么就一直循环弹出队尾值，直到当前元素小于队列的最后一个元素或者队列为空。

一开始想的是在队列中加入数组的数值，但是发现这样不好确定什么时候弹出队列头的值。后来看了别人的做法，发现可以直接存储数组的index。

```java
class Solution {
    //构建一个单调队列
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums == null || nums.length < 2) return nums;
        // 双向队列 保存当前窗口最大值的数组位置 保证队列中数组位置的数值按从大到小排序
        LinkedList<Integer> queue = new LinkedList<>();
        // 结果数组
        int[] result = new int[nums.length-k+1];
        // 遍历nums数组
        for(int i = 0;i < nums.length;i++){
            // 保证整个队列按从大到小排列，新的数一定是从尾部加入，如果当前检验数值大于队尾 那么需要弹出队尾
            while(!queue.isEmpty() && nums[queue.peekLast()] <= nums[i]){
                queue.pollLast();
            }
            // 添加当前值对应的数组下标
            queue.addLast(i);
            // 判断当前队列中队首的值是否有效
            if(queue.peek() <= i-k){
                queue.poll();   
            } 
            // 当窗口长度为k时 保存当前窗口中最大值
            if(i+1 >= k){
                result[i+1-k] = nums[queue.peek()];
            }
        }
        return result;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（k）

#### 解法二：优先队列

构造一个最大堆，当队列容量等于`k`就删除窗口的第一个元素。

```java
class Solution {
    //构建一个单调队列
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums == null || nums.length < 2 || k == 1) return nums;
        // 结果数组
        int[] result = new int[nums.length-k+1];
        // 构造最大堆
        Queue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
            public int compare(Integer a, Integer b) {
                return b-a;
            }
        });
        for(int i = 0; i < nums.length; i++) {
            if(queue.size() == k) queue.remove(nums[i-k]);
            queue.offer(nums[i]);
            if(i >= k-1) result[i-k+1] = queue.peek();
        }
        return result;
    }
}
```

由于删除元素时用的`remove(指定元素)`时间复杂度为O（k），所以整个算法的时间复杂度为O（nk）。超时。

### 241. 为运算表达式设计优先级

给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。你需要给出所有可能的组合的结果。有效的运算符号包含 `+`, `-` 以及 `*` 。

**示例 1:**

```
输入: "2-1-1"
输出: [0, 2]
解释: 
((2-1)-1) = 0 
(2-(1-1)) = 2
```

**示例 2:**

```
输入: "2*3-4*5"
输出: [-34, -14, -10, -10, 10]
解释: 
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10
```

链接：https://leetcode-cn.com/problems/different-ways-to-add-parentheses

#### 解法：分治

对于一个运算式`x op y`，其中`op`是运算操作符，最终的结果就取决于`x`和`y`的数值。所以对于任何一个含有数字和运算符的字符串，都可以通过运算操作符将它分成两个部分，先计算两个部分的值再求最终结果。

```java
class Solution {
    public List<Integer> diffWaysToCompute(String expression) {
        List<Integer> res = new ArrayList<>();
        char[] chars = expression.toCharArray();
        for(int i = 0; i < chars.length; i++) {
            char now = chars[i];
            if(now == '+' || now == '-' || now == '*'){
                List<Integer> left = diffWaysToCompute(expression.substring(0,i));
                List<Integer>right = diffWaysToCompute(expression.substring(i+1));
                for(int l : left) {
                    for(int r : right) {
                        switch(now) {
                            case '+':
                                res.add(l+r);
                                break;
                            case '-':
                                res.add(l-r);
                                break;
                            case '*':
                                res.add(l*r);
                                break;
                        }
                    }
                }
            }
        }
        if(res.size() == 0) res.add(Integer.valueOf(expression));
        return res;
    }
}
```

这题想了很久没有思路，是因为没有想到任何一个式子的实质就是要知道操作符两边的值，从而将一个问题划分成了两个子问题。

### 242. 有效的字母异位词

给定两个字符串 *s* 和 *t* ，编写一个函数来判断 *t* 是否是 *s* 的字母异位词。

**示例 1:**

```
输入: s = "anagram", t = "nagaram"
输出: true
```

**示例 2:**

```
输入: s = "rat", t = "car"
输出: false
```

链接：https://leetcode-cn.com/problems/valid-anagram

#### 解法一：哈希表

最直接的做法就是用一个哈希表记录字母对应的个数，可以通过一个数组来完成。

在开头判断两个字符串的长度是否相等，这样就不用浪费时间判断一些根本不可能是异位词的状况。

```java
class Solution {
    public boolean isAnagram(String s, String t) {
        int lens = s.length();
        int lent = t.length();
        if(lens != lent) return false;
        int[] record = new int[26];
        
        for(int i = 0; i < lens; i++) {
            record[s.charAt(i) - 'a']++;
        }
        for(int i = 0; i < lens; i++) {
            if((--record[t.charAt(i) - 'a']) < 0) {
                return false;
            }
        }
        return true;
    }
}
```

时间复杂度：O（n）

#### 解法二：排序

排序后比较每个位置上的字符是否相等。

```java
class Solution {
    public boolean isAnagram(String s, String t) {
        if(s.length()!= t.length()) return false;
        char[] charS = s.toCharArray();
        char[] charT = t.toCharArray();
        Arrays.sort(charS);
        Arrays.sort(charT);
        for(int i = 0; i < charS.length; i++) {
            if(charS[i] != charT[i]) return false;
        }
        return true;
    }
}
```

时间复杂度：O（nlogn）

#### 解法三：素数相乘

这个思路是我从来没考虑过的，把每个字母都映射到一个素数上，如果最终两个字符串算得结果相同，那么一定是异位词。

但是相乘如果数过大的话会造成溢出，最后结果就不准确了，会出现错误。这题可以使用 `java` 提供的大数类。

```java
import java.math.BigInteger;
class Solution {
    public boolean isAnagram(String s, String t) {
        int[] prime = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101 };
        char[] sArray = s.toCharArray();
        BigInteger sKey = BigInteger.valueOf(1);
        for (int i = 0; i < sArray.length; i++) {
            BigInteger temp = BigInteger.valueOf(prime[sArray[i] - 'a']);
            sKey = sKey.multiply(temp);
        }

        char[] tArray = t.toCharArray();
        BigInteger tKey = BigInteger.valueOf(1);
        for (int i = 0; i < tArray.length; i++) {
            BigInteger temp = BigInteger.valueOf(prime[tArray[i] - 'a']);
            tKey = tKey.multiply(temp);
        }

        return sKey.equals(tKey);
    }
}
```

### 257. 二叉树的所有路径

给定一个二叉树，返回所有从根节点到叶子节点的路径。

**说明:** 叶子节点是指没有子节点的节点。

**示例:**

```
输入:

   1
 /   \
2     3
 \
  5

输出: ["1->2->5", "1->3"]

解释: 所有根节点到叶子节点的路径为: 1->2->5, 1->3
```

链接：https://leetcode-cn.com/problems/binary-tree-paths

#### 解法一：深度优先搜索+回溯

- 如果当前节点是空，直接返回
- 如果当前节点的左右孩子都是空，那么把当前节点加入路径，并添加至答案列表后，就返回
- 继续递归调用当前孩子的左节点，然后右节点。

```java
class Solution {
    List<String> res = new ArrayList<>();
    public List<String> binaryTreePaths(TreeNode root) {
        StringBuilder path = new StringBuilder();
        binaryTreePaths(root, path);
        return res;
    }
    public void binaryTreePaths(TreeNode node, StringBuilder path) {
        if(node == null) return;
        path.append(String.valueOf(node.val));
        if(node.left == node.right) {           
            res.add(path.toString());
            return;
        }else {
            path.append("->");
            binaryTreePaths(node.left, new StringBuilder(path));
            binaryTreePaths(node.right, new StringBuilder(path));
        }
    }
}
```

#### 解法二：广度优先搜索

维护两个队列，一个是节点层序遍历队列，另一个是结果队列。

在每一步迭代中，我们取出队列中的首节点，如果它是叶子节点，则将它对应的路径加入到答案中。如果它不是叶子节点，则将它的所有孩子节点加入到队列的末尾。当队列为空时广度优先搜索结束，我们即能得到答案。

```java
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        if(root == null) return res;
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        Queue<String> pathQueue = new LinkedList<>();
        nodeQueue.offer(root);
        pathQueue.offer(String.valueOf(root.val));
        while(!nodeQueue.isEmpty()) {
            root = nodeQueue.poll();
            StringBuilder path = new StringBuilder(pathQueue.poll());
            if(root.right == root.left) {
                res.add(path.toString());
                continue;
            }else {
                path.append("->");
                if(root.left != null) {
                    nodeQueue.offer(root.left);
                    StringBuilder path1 = new StringBuilder(path);
                    path1.append(String.valueOf(root.left.val));
                    pathQueue.offer(path1.toString());     
                }
                if(root.right != null) {
                    nodeQueue.offer(root.right);
                    StringBuilder path2 = new StringBuilder(path);
                    path2.append(String.valueOf(root.right.val));
                    pathQueue.offer(path2.toString());
                }
            }
        }
        return res;
    }
}
```

### 279. 完全平方数

给定正整数 *n*，找到若干个完全平方数（比如 `1, 4, 9, 16, ...`）使得它们的和等于 *n*。你需要让组成和的完全平方数的个数最少。

给你一个整数 `n` ，返回和为 `n` 的完全平方数的 **最少数量** 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

**示例 1：**

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

**示例 2：**

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```

 链接：https://leetcode-cn.com/problems/perfect-squares/

#### 解法一：动态规划

这题的本质其实就是[背包问题](https://leetcode-cn.com/problems/perfect-squares/solution/yi-pian-wen-zhang-chi-tou-bei-bao-wen-ti-yc1p/)，完全平方数最小为1,最大为sqrt(n),故题目转换为在nums=[1,2.....sqrt(n)]中选任意数平方和为target=n。

```java
class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n + 1]; // 默认初始化值都为0
        for (int i = 1; i <= n; i++) {
            dp[i] = i; // 最坏的情况就是每次+1
            for (int j = 1; j * j <= i; j++) { 
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1); // 动态转移方程
            }
        }
        return dp[n];
    }
}
```

看到有一种 `Static Dynamic Programming`做法。因为在测试的时候如果先测试12，然后又测试13，其实13是可以利用12生成的dp数组的。所以可以声明dp是一个`static`的变量，这样每次调用就不会重复计算了，所有对象将共享 `dp` 。

```java
class Solution {
    static ArrayList<Integer> dp = new ArrayList<>();
    public int numSquares(int n) {
        if(dp.size() == 0) dp.add(0);
        for (int i = dp.size(); i <= n; i++) {
            int min = i;
            for (int j = 1; j * j <= i; j++) { 
                min = Math.min(min, dp.get(i - j * j) + 1); // 动态转移方程
            }
            dp.add(min);
        }
        return dp.get(n);
    }
}
```

#### 解法二：BFS

第一层依次减去一个平方数得到第二层，第二层依次减去一个平方数得到第三层。直到某一层出现了 `0`，此时的层数就是我们要找到平方数和的最小个数。

举个例子，`n = 12`，每层的话每个节点依次减 `1, 4, 9...`。如下图，灰色表示当前层重复的节点，不需要处理。

![img](https://windliang.oss-cn-beijing.aliyuncs.com/279_3.jpg)

```java
class Solution {
    public int numSquares(int n) {
        Queue<Integer> queue = new LinkedList<>();
        HashSet<Integer> visited = new HashSet<>();//存储已经被加入到队列中的数
        queue.offer(n);
        visited.add(n);
        int level = 0;
        while(!queue.isEmpty()) {
            int size = queue.size();
            level++;
            for(int i = 0; i < size; i++) {
                n = queue.poll();
                for(int j = 1; j*j <= n; j++) {
                    if(visited.contains(n-j*j)) continue;
                    if(n - j*j == 0) return level;
                    queue.offer(n-j*j);
                    visited.add(n-j*j);
                }
            }
        }
        return -1;
    }
}
```

#### 解法三：数学思维

[四平方和定理](https://leetcode.wang/[https:/zh.wikipedia.org/wiki/四平方和定理](https:/zh.wikipedia.org/wiki/四平方和定理))，指任何正整数都能表示成四个平方数的和。少于四个平方数的，像 `12` 可以补一个 `0` 也可以看成四个平方数，`12 = 4 + 4 + 4 + 0`。知道了这个定理，对于题目要找的解，其实只可能是 `1, 2, 3, 4` 其中某个数。

[Legendre's three-square theorem](https://en.wikipedia.org/wiki/Legendre's_three-square_theorem) ，这个定理表明，如果正整数 `n` 被表示为三个平方数的和，那么 `n` 不等于 $4^a*(8b+7)$，`a` 和 `b` 都是非负整数。

换言之，如果 $n == 4^a*(8b+7)$，那么他一定不能表示为三个平方数的和，同时也说明不能表示为一个、两个平方数的和，因为如果能表示为两个平方数的和，那么补个 `0`，就能凑成三个平方数的和了。

一个、两个、三个都排除了，所以如果 n == 4^a*(8b+7)*n*==4*a*∗(8*b*+7)，那么 `n` 只能表示成四个平方数的和了。

所以代码的话，我们采取排除的方法。

首先考虑答案是不是 `1`，也就是判断当前数是不是一个平方数。

然后考虑答案是不是 `4`，也就是判断 `n` 是不是等于 4^a*(8b+7)4*a*∗(8*b*+7)。

然后考虑答案是不是 `2`，当前数依次减去一个平方数，判断得到的差是不是平方数。

以上情况都排除的话，答案就是 `3`。

```java
class Solution {
    public int numSquares(int n) {
        //判断答案是否为1
        if(isSquare(n)) return 1;
        //判断答案是否为4
        int tmp = n;
        while(tmp % 4 == 0) {
            tmp /= 4;
        }
        tmp -= 7;
        if(tmp % 8 == 0) return 4;
        //判断答案是否为2
        for(int i = 1; i*i < n; i++) {
            if(isSquare(n - i*i)) return 2;
        }
        //如果不满足上述所有情况，答案只能是3
        return 3;
    }
    private boolean isSquare(int n) {
        int sqrt = (int) Math.sqrt(n);
        return sqrt*sqrt == n;
    }
}
```

### 287. 寻找重复数

给定一个包含 `n + 1` 个整数的数组 `nums` ，其数字都在 `1` 到 `n` 之间（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数** ，找出 **这个重复的数** 。

你设计的解决方案必须不修改数组 `nums` 且只用常量级 `O(1)` 的额外空间。

**示例 1：**

```
输入：nums = [1,3,4,2,2]
输出：2
```

**示例 2：**

```
输入：nums = [3,1,3,4,2]
输出：3
```

**示例 3：**

```
输入：nums = [1,1]
输出：1
```

**示例 4：**

```
输入：nums = [1,1,2]
输出：1
```

链接：https://leetcode-cn.com/problems/find-the-duplicate-number

#### 解法一：快慢指针

可以将数组`nums[i]`看作一个值为`i`的节点所指向的下一个节点值。因为题目规定数组长度为`n+1`，数组里的数不超过`n`，所以必定不会越界。

以上面示例一为例，可以得到下面的环形链表：

![](https://pic.leetcode-cn.com/999e055b41e499d9ac704abada4a1b8e6697374fdfedc17d06b0e8aa10a8f8f6-287.png)

用和142题一样的思路来写即可。

```java
class Solution {
    public int findDuplicate(int[] nums) {
        int slow = 0, fast = 0;
        slow = nums[slow]; //移动一步
        fast = nums[nums[fast]]; //移动两步
        while(slow != fast) {
            slow = nums[slow]; //移动一步
            fast = nums[nums[fast]]; //移动两步
        }
        fast = 0;
        while(slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
}
```

时间复杂度：O（n）

#### 解法二：二分查找

以示例一为例，数组为`[1,3,4,2,2]`，即小于等于`1`的数有`1`个，小于等于`2`的数有`3`个，小于等于`3`的数有4个，小于等于`4`的数有`5`个，即因为多了一个`2`，而小于等于`j`（`j`大于等于`2`）的数都为`j+1`。

可以看做生成了一个数量数组`cnt[i]`代表小于等于`i+1`的数的个数。这个数组一定是单调递增的，容量为`l`，里面的数不会重复，范围从`1`到`l+1`，所以必然会缺少某个数。

又因为题目要求使用的空间复杂度为O（1），所以不能单独用一个数组来存储`cnt`，要在每次查找中计算生成。

这题可以转化为求`cnt`数组中缺少的某个数字，可以通过二分查找法，找小于等于`mid`的个数`cnt[mid-1]`。先猜一个数（有效范围 `[left..right]` 里位于中间的数 `mid`）然后统计原始数组中 **小于等于** `mid` 的元素的个数 `cnt`：

- 如果 `cnt` **严格大于** `mid`。根据抽屉原理，重复元素就在区间 `[left..mid]` 里；
- 否则，重复区间在`[mid+1..right]`里。

继续以上面为例，先找小于等于`2`的个数，计算得到`3`，所以缺少的`cnt`一定小于等于`2`，因为此时的`cnt[mid-1]=mid+1`，`right`指针移到`2`。第二次查找`1`，计算得到`1`，所以缺少的`cnt`一定大于`1`，因为此时的`cnt[mid-1] = mid`，`left`指针移到`2`。左右指针相等跳出循环。

要注意到，题目说，数字可能不止重复一遍，所以只要当`cnt[mid-1] >= mid+1`，`right`指针就要挪动到`mid`。当`cnt[mid-1] <= mid`就要将`left`移动到`mid+1`。

以`[1,2,4,4,4,4]`为例：

```
left	right	mid    cnt
  1	      5      3      2     
  4	      5      4      6
  4       4           
```

可以得到如下代码：

```java
class Solution {
    public int findDuplicate(int[] nums) {
        int left = 1, right = nums.length-1;
        while(left < right) {
            int mid = left + (right-left)/2;
            int cnt = 0;
            for(int i = 0; i < nums.length; i++) {
                if(nums[i] <= mid) cnt++;
            }
            if(cnt <= mid) {
                left = mid+1;
            }else {
                right = mid;
            }
        }
        return left;
    }
}
```

时间复杂度：O（nlogn）

#### 解法三：位运算

如果能知道每一位是0还是1就可以确定出这个重复的数是什么。对于一个长度为`len`的数组，其里面的值只能是从`1`到`len-1`中选取，先计算不重复的`1`到`len-1`，每一位上`1`出现的个数`y`，再计算实际情况该位上`1`出现的个数`x`。

- 如果数组中只有一个`target`出现了两次，其余数都是出现一次。如果`target`在第`i`位上是`1`，那么计算得到的`x`必定是`y+1`，如果是`0`，那么`x=y`。这样就可以得到`target`在每一位上是0还是1。

- 如果数组中的`target`出现了三次及以上，那么在`1`到`len-1`中必定有一个或一些数是缺失的。这个时候相当于用`target` 去替换了这些数，考虑替换对`x`的影响：

  - 如果`target`的第`i`位是1，被替换数该位是0，那么`x>y`
  - 如果`target`的第`i`位是1，被替换数该位是1，那么`x>y`
  - 如果`target`的第`i`位是0，被替换数该位是0，那么`x=y`
  - 如果`target`的第`i`位是0，被替换数该位是1，那么`x<y`

  可以得出，如果第`i`位的`x<=y`，`target`在这一位必然是0，否则是1。

```java
class Solution {
    public int findDuplicate(int[] nums) {
        int bit_max = 31, len = nums.length, res = 0;
        while((len-1) >> bit_max == 0) {
            bit_max--;
        }
        for(int i = 0; i <= bit_max; i++) {
            int x = 0, y = 0;
            int mode = 1 << i;
            for(int j = 0; j < len; j++) {
                if((nums[j] & mode) != 0) x++;
                if((j & mode) != 0) y++;
            }
            if(x > y) res += mode;
        }
        return res;
    }
}
```

时间复杂度：O（nlogn）

### 290. 单词规律

给定一种规律 `pattern` 和一个字符串 `str` ，判断 `str` 是否遵循相同的规律。

这里的 **遵循** 指完全匹配，例如， `pattern` 里的每个字母和字符串 `str` 中的每个非空单词之间存在着双向连接的对应规律。

**示例1:**

```
输入: pattern = "abba", str = "dog cat cat dog"
输出: true
```

**示例 2:**

```
输入:pattern = "abba", str = "dog cat cat fish"
输出: false
```

**示例 3:**

```
输入: pattern = "aaaa", str = "dog cat cat dog"
输出: false
```

**示例 4:**

```
输入: pattern = "abba", str = "dog dog dog dog"
输出: false
```

链接：https://leetcode-cn.com/problems/word-pattern

#### 解法：哈希表

这题就是一个很基本的哈希表的运用，通过pattern里的单个字符映射一个单词。需要注意的是，一定要保证是一对一映射，不能允许同一个单词被多个字符对应。

```java
class Solution {
    public boolean wordPattern(String pattern, String s) {
        String[] ss = s.split(" ");
        if(pattern.length() != ss.length) return false;
        HashMap<Character, String> hm1 = new HashMap<>();
        HashSet<String> hs = new HashSet<>();
        for(int i = 0; i < ss.length; i++) {
            char c = pattern.charAt(i);
            String sn = ss[i];
            if(hm1.containsKey(c)) {
                if(!sn.equals(hm1.get(c))) return false;
            }else {
                if(hs.contains(sn)) return false;
                hm1.put(c,sn);
                hs.add(sn);
            }
        }
        return true;
    }
}
```

### 300. 最长递增子序列

给你一个整数数组 `nums` ，找到其中最长**严格递增**子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**示例 1：**

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```

**示例 2：**

```
输入：nums = [0,1,0,3,2,3]
输出：4
```

**示例 3：**

```
输入：nums = [7,7,7,7,7,7,7]
输出：1
```

链接：https://leetcode-cn.com/problems/longest-increasing-subsequence

#### 解法一：动态规划

数组`dp[i]`表示以`nums[i]`结尾的严格递增子序列的最大长度。

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        if(len <= 1) return len;
        int[] dp = new int[len];
        dp[0] = 1;
        int res = 1;//保留遇到的最大长度
        for(int i = 1; i < len; i++) {
            int max = 0;
            for(int j = i-1; j >= 0; j--) {
                if(nums[i] > nums[j]) max = Math.max(max, dp[j]);
            }
            dp[i] = max+1;
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}
```

时间复杂度：O（n^2^）

#### 解法二：贪心+二分查找

看了这里的[题解](https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/dong-tai-gui-hua-er-fen-cha-zhao-tan-xin-suan-fa-p/)

算是动态规划的另一种变体。考虑的不再是以某个数结尾能够成的最长长度，而是能构成某长度的最小数。

`tail[i]`表示构成长度为`i+1`的严格递增序列末尾的最小值。二分查找插入位置的代码来源于35题。

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        if(len <= 1) return len;
        int[] tail = new int[len];
        Arrays.fill(tail,2501);
        tail[0] = nums[0];
        int res = 1;
        for(int i = 1; i < len; i++) {
            if(nums[i] > tail[res-1]) {
                tail[res++] = nums[i];
            }else if(nums[i] < tail[res-1]) {
                //二分查找要插入的位置
                tail[searchInsert(tail, nums[i])] = nums[i];
            }
        }
        return res;
    }

    
    private int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        if(right == 0) return target > nums[0] ? 1 : 0;
        if(right == -1) return 0;
        
        while(left <= right) {
            int mid = left + (right - left)/2;
            if(nums[mid] == target) return mid;
            if(nums[mid] < target) {
                left = mid + 1;
            }else{
                right = mid - 1;
            }
        }
        return left;
    }
}
```

时间复杂度：O（nlogn）

### 301. 删除无效的括号

给你一个由若干括号和字母组成的字符串 `s` ，删除最小数量的无效括号，使得输入的字符串有效。返回所有可能的结果。答案可以按 **任意顺序** 返回。

**示例 1：**

```
输入：s = "()())()"
输出：["(())()","()()()"]
```

**示例 2：**

```
输入：s = "(a)())()"
输出：["(a())()","(a)()()"]
```

**示例 3：**

```
输入：s = ")("
输出：[""]
```

#### 解法一：深度优先搜索

最暴力的方法是每个括号都尝试一下保留或删除。但是这样的时间效率太低。那么可不可以提前知道需要删除括号的最小数量呢？答案是可以的，只需要通过一轮遍历即可。用两个索引`left`和`right`分别保存多余的左括号和右括号数量。

- 从左往右遍历，如果遇到左括号，`left+1`
- 如果遇到右括号，若有多余未配对的`left`，就`left-1`，如果`left=0`，那么`right+1`

```
0 1 2 3 4 5 6 7
( ) ) ) ( ( ( )  
i = 0, left = 1, right = 0
i = 1, left = 0, right = 0
i = 2, left = 0, right = 1
i = 3, left = 0, right = 2
i = 4, left = 1, right = 2
i = 5, left = 2, right = 2
i = 6, left = 3, right = 2
i = 7, left = 2, right = 2
```

写一个回溯函数，删除掉所有多余括号后判断是否为有效的括号且没有被添加过，如果是就加入结果。

```java
class Solution {
    List<String> res = new ArrayList<>();
    public List<String> removeInvalidParentheses(String s) {
        int left = 0, right = 0;
        char[] array = s.toCharArray();
        for(int i = 0; i < array.length; i++) {
            if(array[i] == '(') {
                left++;
            }else if(array[i] == ')') {
                if(left > 0) {
                    left--;
                }else {
                    right++;
                }
            }
        }
        removeInvalidParentheses(array, 0, left, right, new StringBuilder());
        return res;
    }
    private void removeInvalidParentheses(char[] array, int start, int left, int right, StringBuilder exp) {
        for(int i = start; i < array.length; i++) {
            if(left > 0 && array[i] == '(') {
                removeInvalidParentheses(array, i+1, left-1, right, new StringBuilder(exp));
            }
            if(right > 0 && array[i] == ')') {
                removeInvalidParentheses(array, i+1, left, right-1, new StringBuilder(exp));
            }
            exp.append(array[i]);
        }
        if(left == 0 && right == 0) {
            String t = exp.toString();
            if(isValid(t) && (!res.contains(t))) res.add(t);
        }
    }
    private boolean isValid(String t) {        
        int left = 0, right = 0;
        char[] array = t.toCharArray();
        for(int i = 0; i < array.length; i++) {
            if(array[i] == '(') {
                left++;
            }else if(array[i] == ')') {
                if(left > right) {
                    left--;
                }else {
                    right++;
                }
            }
        }
        return left == 0 && right == 0;
    }
}
```

这样写有一个很明显的弊端，就是在删除完括号后还需要再进行一次判断是否为有效的字符串。其实这一步可以在回溯函数中完成，只要每一步都确保当前的右括号数没有大于左括号数即可。

```java
class Solution {
    List<String> res = new ArrayList<>();
    public List<String> removeInvalidParentheses(String s) {
        int left = 0, right = 0;
        char[] array = s.toCharArray();
        for(int i = 0; i < array.length; i++) {
            if(array[i] == '(') {
                left++;
            }else if(array[i] == ')') {
                if(left > 0) {
                    left--;
                }else {
                    right++;
                }
            }
        }
        removeHelp(array, 0, 0, 0, left, right, new StringBuilder());
        return res;
    }
    private void removeHelp(char[] array, int start, int leftCount, int rightCount, int leftRemove, int rightRemove, StringBuilder exp) {        
        if(start == array.length) {
            if(leftRemove == 0 && rightRemove == 0) {
                String t = exp.toString();
                if(!res.contains(t)) res.add(t);
            }            
            return;
        }
        char c = array[start];
        if(leftRemove > 0 && c == '(') removeHelp(array, start+1, leftCount, rightCount, leftRemove-1, rightRemove, exp);
        if(rightRemove > 0 && c == ')') removeHelp(array, start+1, leftCount, rightCount, leftRemove, rightRemove-1, exp);
        exp.append(c);
        if(c == '(') {
            removeHelp(array, start+1, leftCount+1, rightCount, leftRemove, rightRemove, exp);
        }else if(c == ')') {
            if(rightCount < leftCount) removeHelp(array, start+1, leftCount, rightCount+1, leftRemove, rightRemove, exp);
        }else {
            removeHelp(array, start+1, leftCount, rightCount, leftRemove, rightRemove, exp);
        }
        exp.deleteCharAt(exp.length() - 1);
    }
}
```

#### 解法二：广度优先搜索

把所有上一层level中的每个元素都拿出来，列举出在删除一个括号后的所有可能的情况。(不管删除以后是否合法），添加到下一个level中的元素。

```java
public class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        if (s == null) {
            return res;
        }
        // 广度优先遍历须要的队列和防止重复遍历的哈希表 visited 
        Set<String> visited = new HashSet<>();
        visited.add(s);
        Queue<String> queue = new LinkedList<>();
        queue.add(s);
        // 找到目标值以后推出循环
        boolean found = false;
        while (!queue.isEmpty()) {
            // 最优解一定在同一层
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String front = queue.poll();
                if (isValid(front)) {
                    res.add(front);
                    found = true;
                }

                int currentWordLen = front.length();
                char[] charArray = front.toCharArray();
                for (int j = 0; j < currentWordLen; j++) {
                    if (front.charAt(j) != '(' && front.charAt(j) != ')') {
                        continue;
                    }

                    // 注意 new String() 方法的 API，第 1 个参数是字符数组，第 2 个参数是字符数组的起始下标，第 3 个参数是截取的字符的长度
                    String next = new String(charArray, 0, j) + new String(charArray, j + 1, currentWordLen - j - 1);
                    if (!visited.contains(next)) {
                        queue.offer(next);
                        visited.add(next);
                    }
                }
            }

            // 这一层找到以后，退出外层循环，返回结果
            if (found) {
                break;
            }
        }
        return res;
    }

    public boolean isValid(String s) {
        char[] charArray = s.toCharArray();
        int count = 0;
        for (char c : charArray) {
            if (c == '(') {
                count++;
            } else if (c == ')') {
                count--;
            }
            if (count < 0) {
                return false;
            }
        }
        return count == 0;
    }
}
```

### 303. 区域和检索-数组不可变

给定一个整数数组  `nums`，求出数组从索引 `i` 到 `j`（`i ≤ j`）范围内元素的总和，包含 `i`、`j `两点。

实现 `NumArray` 类：11

- `NumArray(int[] nums)` 使用数组 `nums` 初始化对象
- `int sumRange(int i, int j)` 返回数组 `nums` 从索引 `i `到 `j`（i ≤ j）范围内元素的总和，包含 `i`、`j `两点（也就是 `sum(nums[i], nums[i + 1], ... , nums[j])`）

**示例：**

```
输入：
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
输出：
[null, 1, -1, -3]

解释：
NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
numArray.sumRange(0, 2); // return 1 ((-2) + 0 + 3)
numArray.sumRange(2, 5); // return -1 (3 + (-5) + 2 + (-1)) 
numArray.sumRange(0, 5); // return -3 ((-2) + 0 + 3 + (-5) + 2 + (-1))
```

链接：https://leetcode-cn.com/problems/range-sum-query-immutable

#### 解法：前缀和

最简单的方法是每次sumRange方法计算和，但是这并不是最优的。采用前缀和的方式来解题，前缀和的应用之前也见到过的，但在这题并没有迅速想到，说明对于前缀和的应用还不是很熟悉。

```java
class NumArray {
    private int[] array;
    public NumArray(int[] nums) {
        array = new int[nums.length];
        for(int i = 0; i < nums.length; i++) {
            array[i] = (i>0 ? array[i-1] : 0) + nums[i];
        }
    }
    
    public int sumRange(int left, int right) {
        return array[right] - (left >= 1 ? array[left-1] : 0);
    }
}
```

### 309. 最佳买卖股票时机含冷冻期

给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

- 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

- 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

  **示例**:

```
输入: [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown

#### 解法：动态规划

买卖股票也是老朋友了，这里不同的是规定了一个冷冻期，即必须要在卖出后至少一天才能再次买入。

依然规定数组`dp[i][0]`表示第i天持有股票所能获得的最大收益，`dp[i][1]`表示第i天不持有股票所能获得的最大收益。在计算`dp[i][0]`的时候如果要在第`i`天买入，那么必须在第`i-2`天卖出的状态上减去第`i`天的股票价格。

```java
class Solution {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        if(len <= 1) return 0;
        //dp[i][0]表示第i天持有股票：可能是当天买入，也可能是之前买入
        //dp[i][1]表示第i天不持有股票：可能是当天卖出，也可能是之前卖出
        int[][] dp = new int[len][2];
        dp[0][0] = -prices[0];
        for(int i = 1; i < len; i++) {
            dp[i][0] = Math.max(dp[i-1][0], (i >= 2 ? dp[i-2][1] : 0) - prices[i]);
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] + prices[i]);
        }
        return dp[len-1][1];
    }
}
```

### 322. 零钱兑换

给定不同面额的硬币 `coins` 和一个总金额 `amount`。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 `-1`。你可以认为每种硬币的数量是无限的。

**示例 1：**

```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

**示例 2：**

```
输入：coins = [2], amount = 3
输出：-1
```

**示例 3：**

```
输入：coins = [1], amount = 0
输出：0
```

**示例 4：**

```
输入：coins = [1], amount = 1
输出：1
```

**示例 5：**

```
输入：coins = [1], amount = 2
输出：2
```

链接：https://leetcode-cn.com/problems/coin-change

#### 解法：完全背包动态规划

完全背包对应的是每个元素可以重复选择的情况。题目要求构成`amount`的最小硬币个数，构造一个`dp[i]`数组表示凑成`i`的最小硬币个数。

- 硬币`i`存在`coins`数组中，那么答案为1
- 硬币`i`不在`coins`数组中，但是硬币`i-j`在数组中，答案为`dp[i-j]+1`

然后问题来了，难道每确定一个`dp[i]`都要寻找从`1`到`i`的硬币是否在`coins`中吗？其实不用，只需要改一下遍历的顺序即可。按照原思路是从`1`到`amount`确定`dp[i]`，但是可以用两层循环，外循环`[coins[0]..coins[n]]`，内循环`[dp[1]..dp[amount]]`。

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        if(amount == 0) return 0;
        int[] dp = new int[amount+1];
        for(int i = 1; i <= amount; i++) dp[i] = -1;//除了dp[0]初始化0，其他值都初始化为-1
        for(int i = 0; i < coins.length; i++) {
            //外层从coins中选一个元素
            for(int j = coins[i]; j <= amount; j++) {
                //内层遍历不同的容量
                if(dp[j-coins[i]] != -1) {
                    dp[j] = Math.min(dp[j-coins[i]] + 1, (dp[j] <= 0 ? 11111 : dp[j]));
                } 
            }
        }
        return dp[amount];
    }
}
```

时间复杂度：O（mn）

### 332. 重新安排行程

给定一个机票的字符串二维数组 `[from, to]`，子数组中的两个成员分别表示飞机出发和降落的机场地点，对该行程进行重新规划排序。所有这些机票都属于一个从 `JFK`（肯尼迪国际机场）出发的先生，所以该行程必须从 `JFK` 开始。

**提示：**

1. 如果存在多种有效的行程，请你按字符自然排序返回最小的行程组合。例如，行程 `["JFK", "LGA"]` 与 `["JFK", "LGB"]` 相比就更小，排序更靠前
2. 所有的机场都用三个大写字母表示（机场代码）
3. 假定所有机票至少存在一种合理的行程
4. 所有的机票必须都用一次 且 只能用一次

**示例 1：**

```
输入：[["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
输出：["JFK", "MUC", "LHR", "SFO", "SJC"]
```

**示例 2：**

```
输入：[["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
输出：["JFK","ATL","JFK","SFO","ATL","SFO"]
解释：另一种有效的行程是 ["JFK","SFO","ATL","JFK","ATL","SFO"]。但是它自然排序更大更靠后。
```

链接：https://leetcode-cn.com/problems/reconstruct-itinerary

#### 解法一：朴素回溯法

因为要返回自然排序最小的行程组合，所以可以先给机票数组排序，然后用回溯法，最先找到的把所有机票都用掉的行程组合就是答案。为了防止多次利用同一张机票，需要在创建一个布尔数组用于标记某张机票是否已经被使用过。

```java
class Solution {
    List<String> res;
    public List<String> findItinerary(List<List<String>> tickets) {
        Collections.sort(tickets, new Comparator<List<String>>() {
            public int compare(List<String> l1, List<String> l2) {
                int flag =l1.get(0).compareTo(l2.get(0));
                if(flag == 0) flag =l1.get(1).compareTo(l2.get(1));
                return flag;
            }
        });
        res = new ArrayList<>();
        res.add("JFK");
        boolean[] used = new boolean[tickets.size()];
        dfs("JFK", tickets, used);
        return res;
    }
    public boolean dfs(String from, List<List<String>> tickets, boolean[] used){
        if(res.size() == tickets.size()+1) return true;
        for(int i = 0; i < tickets.size(); i++) {
            List<String> tmp = tickets.get(i);
            if(!used[i] && tmp.get(0).compareTo(from) == 0) {
                res.add(tmp.get(1));
                used[i] = true;
                if(dfs(tmp.get(1), tickets, used)) return true;
                res.remove(res.size()-1);
                used[i] = false;
            }
        }
        return false;
    }
}
```

#### 解法二：哈希+TreeMap+回溯

本质与上面是相同的，只不过在排序上用到了一个`TreeMap`结构，在防重复利用方面使用的是键值对中存储数字。

构造一个`HashMap<String, Map<String, Integer>> map`，里层的`Map<String, Integer>`类型是`TreeMap`，与普通的Map不同的是它按排序顺序（自然顺序）或按`Comparator`创建时提供的密钥存储密钥。哈希表的`key`存储出发地点`src`，` Map<String, Integer>`中的`key`存储到达地`des`，`value`存储的是从`src`到`des`这张机票的张数。

```java
class Solution {
    private Deque<String> res;
    private Map<String, Map<String, Integer>> map;

    private boolean backTracking(int ticketNum){
        if(res.size() == ticketNum + 1){
            return true;
        }
        String last = res.getLast();
        if(map.containsKey(last)){//防止出现null
            for(Map.Entry<String, Integer> target : map.get(last).entrySet()){
                int count = target.getValue();
                if(count > 0){
                    res.add(target.getKey());
                    target.setValue(count - 1);
                    if(backTracking(ticketNum)) return true;
                    res.removeLast();
                    target.setValue(count);
                }
            }
        }
        return false;
    }

    public List<String> findItinerary(List<List<String>> tickets) {
        map = new HashMap<String, Map<String, Integer>>();
        res = new LinkedList<>();
        for(List<String> t : tickets){
            Map<String, Integer> temp;
            if(map.containsKey(t.get(0))){
                temp = map.get(t.get(0));
                temp.put(t.get(1), temp.getOrDefault(t.get(1), 0) + 1);
            }else{
                temp = new TreeMap<>();//升序Map
                temp.put(t.get(1), 1);
            }
            map.put(t.get(0), temp);

        }
        res.add("JFK");
        backTracking(tickets.size());
        return new ArrayList<>(res);
    }
}
```

#### 解法三：Hierholzer 算法

这一题的本质是求有向图的欧拉路径：指存在一个顶点，从它出发，沿着有向边的方向，可以不重复地遍历图中所有的边。 

由于题目要求是排序小的为结果，所以可以用一个优先队列来存储从`src`出发的所有`des`。每次到达一个地点，都将该`des`从优先队列中拉取删除。

但是对于下图的情况：

![](https://assets.leetcode-cn.com/solution-static/332/332_fig2.png)

当我们先访问 AAA 时，我们无法回到 JFK，这样我们就无法访问剩余的边了。

也就是说，当我们贪心地选择字典序最小的节点前进时，我们可能先走入「死胡同」，从而导致无法遍历到其他还未访问的边。于是我们希望能够遍历完当前节点所连接的其他节点后再进入「死胡同」。

> Hierholzer 算法用于在连通图中寻找欧拉路径，其流程如下：
>
> 1. 从起点出发，进行深度优先搜索。
> 2. 每次沿着某条边从某个顶点移动到另外一个顶点的时候，都需要删除这条边。
> 3. 如果没有可移动的路径，则将所在节点加入到栈中，并返回。

```java
class Solution {
    LinkedList<String> result;
    Map<String,PriorityQueue<String>> map;
    public List<String> findItinerary(List<List<String>> tickets) {
        //初始化
        this.result = new LinkedList<>();
        this.map = new HashMap<>();
        for(List<String> ticket:tickets){
            String from = ticket.get(0);
            String to = ticket.get(1);

            PriorityQueue<String> destinations = map.get(from);

            if(destinations == null){
                destinations = new PriorityQueue<String>();
                map.put(from,destinations);
            }
            //使用优先队列来保持目的地从小到大的字典序
            destinations.add(to);
        }
        dfs("JFK");
        return result;
    }
    public void dfs(String from){
        PriorityQueue<String> arrivals = map.get(from);
        while(arrivals != null && !arrivals.isEmpty()){
            dfs(arrivals.remove());
        }
        result.addFirst(from);//先递归再将节点加入数组中
    }
}
```

### 337. 打家劫舍 III

在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

**示例 1:**

    输入: [3,2,3,null,3,null,1]
         3
        / \
       2   3
        \   \ 
         3   1
    
    输出: 7 
    解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
   **示例 2:**

    输入: [3,4,5,1,3,null,1]     
         3
        / \
       4   5
      / \   \ 
     1   3   1
    
    输出: 9
    解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.

链接：https://leetcode-cn.com/problems/house-robber-iii

#### 解法一：暴力递归

每一棵树如果偷根节点，那么它的左右孩子就不能偷，如果不偷根节点，就可以偷左右孩子。得到如下递归函数

```java
class Solution {
    public int rob(TreeNode root) {
        if(root == null) return 0;
        int cur = root.val;
        if(root.left != null) {
            cur += rob(root.left.left) + rob(root.left.right);
        }
        if(root.right != null) {
            cur += rob(root.right.left) + rob(root.right.right);
        }
        return Math.max(cur, rob(root.left)+rob(root.right));
    }
}
```

但是超出了时间限制，这是因为会做重复递归，可以用memorization的方法。

#### 解法二：记忆+递归

记录下偷每个节点能获得的最大收益。

```java
class Solution {
    HashMap<TreeNode, Integer> memo = new HashMap<>();
    public int rob(TreeNode root) {
        if(root == null) return 0;
        if(memo.containsKey(root)) return memo.get(root);
        int cur = root.val;
        if(root.left != null) {
            cur += rob(root.left.left) + rob(root.left.right);
        }
        if(root.right != null) {
            cur += rob(root.right.left) + rob(root.right.right);
        }
        cur = Math.max(cur, rob(root.left)+rob(root.right));
        memo.put(root, cur);
        return cur;
    }
}
```

#### 解法三：再优化

其实每个节点其本质就是偷与不偷所能获得的最大收益，设计一个能一并返回偷与不偷的最大收益的函数即可。

```java
class Solution {
    public int rob(TreeNode root) {
        int[] res = robTree(root);
        return Math.max(res[0],res[1]);
    }
    public int[] robTree(TreeNode root) {
        if(root == null) return new int[] {0,0};
        int[] left = robTree(root.left);
        int[] right = robTree(root.right);
        //如果要抢该节点，就把不抢左右子节点的财产加起来再加上本节点的财产
        int qiang = left[1] + right[1] + root.val;
        //如果不抢这个节点，则选取左右节点中分别最大的财产相加
        int buqiang = Math.max(left[0], left[1]) + Math.max(right[0],right[1]);
        return new int[] {qiang, buqiang};
    }
}
```

### 343. 整数拆分

给定一个正整数 *n*，将其拆分为**至少**两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

**示例 1:**

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。
```

**示例 2:**

```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
```

**说明:** 你可以假设 n 不小于 2 且不大于 58。
链接：https://leetcode-cn.com/problems/integer-break

#### 解法一：动态规划

每个数能够拆分得到的最大乘积取决于前面的数，所以可以用动态规划来写。`dp[i]`是最大乘积，拆分成`j`和`i-j`两部分，`dp[i]`可能等于`dp[j]*(i-j)`也可能等于`j*(i-j)`。比如，`dp[2]=1`,`dp[3]=1*2`而不是`dp[3]=1*dp[2]=1`。

```java
class Solution {
    public int integerBreak(int n) {
        int[] dp = new int[n+1];
        dp[2] = 1;
        for(int i = 3; i <= n; i++) {
            for(int j = 2; j < i; j++) {
                dp[i] = Math.max(dp[i], Math.max(dp[j], j)*(i-j));
            }
        }
        return dp[n];
    }
}
```

时间复杂度：O（n^2^）

#### 解法二：数学思想

具体证明在[这](https://leetcode-cn.com/problems/integer-break/solution/343-zheng-shu-chai-fen-tan-xin-by-jyd/)

拆分规则：
最优： 3 。把数字 nn 可能拆为多个因子 3 ，余数可能为 0,1,2 三种情况。
次优： 2 。若余数为 2；则保留，不再拆为 1+1 。
最差： 1 。若余数为 1 ；则应把一份 3 + 1 替换为 2 + 2，因为 2* 2 > 3 * 1

```java
class Solution {
    public int integerBreak(int n) {
        if(n <= 3) return n-1;
        int a = n/3, b = n - 3*a;
        if(b == 0) return (int)Math.pow(3,a);
        if(b == 1) return (int)Math.pow(3,a-1)*4;
        return (int)Math.pow(3,a)*2;
    }
}
```

不用库函数pow的写法：

```java
class Solution {
    public int integerBreak(int n) {
        if(n==2) return 1;
        if(n==3) return 2;
        if(n==4) return 4;
        int res = 1;
        while(n > 4) {
            res *= 3;
            n -= 3;
        }
        res *= n;
        return res;
    }
}
```

时间复杂度：O（n）

### 344. 反转字符串

编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 `char[]` 的形式给出。不要给另外的数组分配额外的空间，你必须**[原地](https://baike.baidu.com/item/原地算法)修改输入数组**、使用 O(1) 的额外空间解决这一问题。你可以假设数组中的所有字符都是 [ASCII](https://baike.baidu.com/item/ASCII) 码表中的可打印字符。

**示例 1：**

```
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]
```

**示例 2：**

```
输入：["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]
```

链接：https://leetcode-cn.com/problems/reverse-string

#### 解法：

比较简单直接上代码了

```java
class Solution {
    public void reverseString(char[] s) {
        int left = 0, right = s.length - 1;
        if(right <= 0) return;
        while(left < right) {
            swap(s,left++,right--);
        }
    }
    private void swap(char[] s, int i, int j) {
        char tmp = s[i];
        s[i] = s[j];
        s[j] = tmp;
    }
}
```

因为数组所有值都可以用ASCII码表示，所以可以用异或运算交换，提高速度。

```java
class Solution {
    public void reverseString(char[] s) {
        int left = 0, right = s.length - 1;
        if(right <= 0) return;
        while(left < right) {
            swap(s,left++,right--);
        }
    }
    private void swap(char[] s, int i, int j) {
        s[i] ^= s[j];
        s[j] ^= s[i];
        s[i] ^= s[j];
    }
}
```

时间复杂度：O（n）

### 345. 反转字符串中的元音字母

编写一个函数，以字符串作为输入，反转该字符串中的元音字母。

**示例 1：**

```
输入："hello"
输出："holle"
```

**示例 2：**

```
输入："leetcode"
输出："leotcede"
```

 链接：https://leetcode-cn.com/problems/reverse-vowels-of-a-string/

####  解法：

这题有个坑的地方在于，可能是大写的元音字母。可以用一个哈希表存储所有元音字母。

```java
class Solution {
    private final static HashSet<Character> vowels = new HashSet<>(
        Arrays.asList('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'));
    //a、e、i、o、u
    public String reverseVowels(String s) {
        if(s.length() <= 1) return s;
        char[] array = s.toCharArray();
        int l = 0, r = array.length - 1;
        while(l < r) {
            if(vowels.contains(array[l]) && vowels.contains(array[r])) {
                swap(array,l,r);
                l++;
                r--;
            }else if(vowels.contains(array[l])) {
                r--;
            }else if(vowels.contains(array[r])) {
                l++;
            }else {
                l++;
                r--;
            }
        }
        return String.valueOf(array);
    }
    private void swap(char[] a, int i, int j) {
        char tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}
```

时间复杂度：O(n)

### 347. 前K个高频元素

给你一个整数数组`nums`和一个整数`k`，请你返回其中出现频率前`k`高的元素。你可以按 **任意顺序** 返回答案。

 **示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

#### 解法一：哈希表

用一个哈希表存储数字对应的出现频次。

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        int[] res = new int[k];    // 结果数组
        Map<Integer, Integer> map = new HashMap();
        // 统计数组中各元素出现的次数
        for(int num : nums){
            if(map.containsKey(num)){
                map.put(num, map.get(num) + 1);
            }else{
                map.put(num, 1);
            }
        }

        int maxTimes = 0;    // 出现最多的元素的出现次数
        // 找出出现次数最多的元素出现的次数
        for(Map.Entry<Integer, Integer> entry : map.entrySet()){
            if(entry.getValue() > maxTimes){
                maxTimes = entry.getValue();
            }
        }

        // 按出现次数从大到小添加到结果数组
        while(k > 0){
            for(Map.Entry<Integer, Integer> entry : map.entrySet()){
                if(entry.getValue() == maxTimes){
                    res[k - 1] = entry.getKey();
                    k--;
                }
            }
            maxTimes--;
        }

        return res;
    }
}
```

时间复杂度：O（k*n）

#### 解法二：优先队列

遇到**前k个大**这样的题好像都可以用。

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        if(nums.length == 1) return nums;
        Arrays.sort(nums);
        int prev = nums[0];
        int count = 1;
        int[] res = new int[k];
        Queue<int[]> que = new PriorityQueue<>((o1,o2)-> o2[1]-o1[1]);
        for(int i = 1; i < nums.length; i++) {
            if(nums[i] == prev) {
                count++;
            }else {                
                que.add(new int[] {prev, count});
                count = 1;
                prev = nums[i];                
            }
        }
        que.add(new int[] {prev, count});
        for(int j = 0; j < k; j++) {
            res[j] = que.poll()[0];
        }        
        return res;
    }
}
```

#### 解法三：快速排序

先统计频次，用一个`List<int[]>`存放对应元素和频次，根据频次来对该结构快速排序（从大到小），如果当前分割元素最后所在位置`= k`可以不再进行后续操作，如果`< k`还要对该元素后面的部分进行快速排序，如果`>k`要对该元素前面部分进行快速排序。

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        int[] res = new int[k];    // 结果数组
        Map<Integer, Integer> map = new HashMap();
        // 统计数组中各元素出现的次数
        for(int num : nums){
            if(map.containsKey(num)){
                map.put(num, map.get(num) + 1);
            }else{
                map.put(num, 1);
            }
        }
        ArrayList<int[]> list = new ArrayList<>();        
        // 将哈希表中的结果存放到数组中
        for(Map.Entry<Integer, Integer> entry : map.entrySet()){
            int[] tmp = new int[2];
            tmp[0] = entry.getKey();
            tmp[1] = entry.getValue();
            list.add(tmp);
        }
        //快速排序
        if(list.size() > k) quickSort(list, 0, list.size()-1, k);
        // 按出现次数从大到小添加到结果数组
        for(int i = 0; i < k; i++) {
            res[i] = list.get(i)[0];
        }
        return res;
    }
    private void quickSort(List<int[]> list, int start, int end, int k) {
        if(end <= start) return;
        //生成随机分割点
        Random random = new Random();
        int pos = random.nextInt(end - start) + start;
        int pivot = list.get(pos)[1];
        Collections.swap(list, start, pos);
        int i = start, j = end+1;
        while(true) {
            while(list.get(++i)[1] > pivot) {
                if(i == end) break;
            }
            while(list.get(--j)[1] < pivot) {
                if(j == start) break;
            }
            if(i >= j) break;
            Collections.swap(list, i, j);
        }
        Collections.swap(list, start, j);
        //如果分割点最终所在的位置是k或者k-1，可以保证整个数组的前k个数频率最高
        if(j == k || j == k-1) return;
        if(j > k) quickSort(list, start, j-1, k);
        quickSort(list, j+1, end, k);
    }
}
```

时间复杂度：原版的快速排序算法的平均时间复杂度为 O(NlogN)。我们的算法中，每次只需在其中的一个分支递归即可，因此算法的平均时间复杂度降为 O(N)。设处理长度为 N 的数组的时间复杂度为 f(N)。由于处理的过程包括一次遍历和一次子分支的递归，最好情况下，有 f(N) = O(N) + f(N/2)，根据 [主定理](https://baike.baidu.com/item/%E4%B8%BB%E5%AE%9A%E7%90%86/3463232)，能够得到 f(N) = O(N)。最坏情况下，每次取的中枢数组的元素都位于数组的两端，时间复杂度退化为 O(N^2^)。但由于我们在每次递归的开始会先随机选取中枢元素，故出现最坏情况的概率很低。
平均情况下，时间复杂度为 O(N)。

### 349. 两个数组的交集

给定两个数组，编写一个函数来计算它们的交集。

**示例 1：**

```
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]
```

**示例 2：**

```
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]
```

链接：https://leetcode-cn.com/problems/intersection-of-two-arrays

#### 解法：哈希集合

一个简单的哈希集合的应用。

```java
class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set = new HashSet();
        Set<Integer> intersection = new HashSet();
        for(int num : nums1) {
            set.add(num);
        }
        for(int num : nums2) {
            if(set.contains(num)) intersection.add(num);
        }
        int size = intersection.size();
        int[] res = new int[size];
        int i = 0;
        for(int num : intersection) {
            res[i++] = num;
        }
        return res;
    }
}
```

### 350. 两个数组的交集 II

给定两个数组，编写一个函数来计算它们的交集。

**示例 1：**

```
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]
```

**示例 2:**

```
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]
```

链接：https://leetcode-cn.com/problems/intersection-of-two-arrays-ii

#### 解法：哈希表

因为涉及到数出现的频次，所以要用哈希表键值对来储存。

```java
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        int length1 = nums1.length, length2 = nums2.length, p = 0;
        if(length1 > length2) return intersect(nums2, nums1);
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int[] intersect = new int[length1];
        for(int i = 0; i < length1; i++) {
            map.put(nums1[i],map.getOrDefault(nums1[i],0) + 1);            
        }
        for(int i = 0; i < length2; i++) {
            int times = map.getOrDefault(nums2[i],0);
            //当这个数值=0时，可能是哈希表中根本没有nums[2]，也可能是nums[2]这个数在数组2中出现的次数大于数组1中的频次。
            if(times > 0) {
                map.put(nums2[i], times - 1);
                intersect[p++] = nums2[i];
            }else {
                continue;
            }            
        }
        return Arrays.copyOfRange(intersect, 0, p);
    }
}
```

代码中如果数组1长度大于数组2长度，就反过来调用本函数的小技巧非常巧妙。

### 376. 摆动序列

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 **摆动序列 。**第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

- 例如， `[1, 7, 4, 9, 2, 5]` 是一个 **摆动序列** ，因为差值 `(6, -3, 5, -7, 3)` 是正负交替出现的。
- 相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。


**子序列** 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。给你一个整数数组 `nums` ，返回 `nums` 中作为 **摆动序列** 的 **最长子序列的长度** 。

**示例 1：**

```
输入：nums = [1,7,4,9,2,5]
输出：6
解释：整个序列均为摆动序列，各元素之间的差值为 (6, -3, 5, -7, 3) 。
```

**示例 2：**

```
输入：nums = [1,17,5,10,13,15,10,5,16,8]
输出：7
解释：这个序列包含几个长度为 7 摆动序列。
其中一个是 [1, 17, 10, 13, 10, 16, 8] ，各元素之间的差值为 (16, -7, 3, -3, 6, -8) 。
```

**示例 3：**

```
输入：nums = [1,2,3,4,5,6,7,8,9]
输出：2
```

链接：https://leetcode-cn.com/problems/wiggle-subsequence

#### 解法一：贪心+空间

用一个数组`list`来存储已经构成的摆动序列。按照`nums`数组从前往后遍历的顺序，如果当前数大于（小于）`list`的末尾值，且`list`末尾数小于（大于）前一个数，就可以将当前值加入`list`，否则就要替换`list`末尾值。

```java
class Solution {
    public int wiggleMaxLength(int[] nums) {
        int len = nums.length;
        if(len <= 1) return len;
        ArrayList<Integer> list = new ArrayList<>();
        for(int i = 0; i < len; i++) {
            if(i == 0) {
                list.add(nums[i]);
            }else {
                int size = list.size();
                boolean flag = (nums[i]>list.get(size-1) && (size<=1 || list.get(size-1)<list.get(size-2))) || (nums[i]<list.get(size-1) && (size<=1 || list.get(size-1)>list.get(size-2)));
                if(flag) {
                    list.add(nums[i]);
                }else {
                    list.set(size-1,nums[i]);
                }
            }
        }
        return list.size();
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

空间上是可以优化的，可以只用一个数存储之前的差值与上一个数。

#### 解法二：贪心

用一个布尔值`postive`表示之前的差值是否为正数，再用一个数`prev`存储摆动数组的尾部。

```java
class Solution {
    public int wiggleMaxLength(int[] nums) {
        int len = nums.length;
        if(len <= 1) return len;
        int prev = nums[0];
        int cnt = 1;
        boolean positive = true;
        for(int i = 1; i < len; i++) {
            if(nums[i] == prev) continue;
            if(cnt > 1) {
                if((nums[i] > prev && !positive) || (nums[i] < prev && positive)) {
                    positive = !positive;
                    cnt++;
                }
            }else {
                positive = nums[i] - prev > 0;
                cnt++;
            }
            prev = nums[i];
        }
        return cnt;
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 377. 组合总和 IV

给你一个由 **不同** 整数组成的数组 `nums` ，和一个目标整数 `target` 。请你从 `nums` 中找出并返回总和为 `target` 的元素组合的个数。题目数据保证答案符合 32 位整数范围。

**示例 1：**

```
输入：nums = [1,2,3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。
```

**示例 2：**

```
输入：nums = [9], target = 3
输出：0
```

链接：https://leetcode-cn.com/problems/combination-sum-iv

#### 解法一：递归

最开始很直观的想法就是递归，但是一尝试发现，超时了。在[1,2,3]，target=32这个测试样例上过不去。

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        if(target == 0) return 1;
        if(target < 0) return 0;
        int ans = 0;
        for(int i = 0; i < nums.length; i++) {
            ans += combinationSum4(nums, target-nums[i]);
        }
        return ans;
    }
}
```

#### 解法二：动态规划

用`dp[i]`表示和为`i`的方案数。用两层循环，外循环是从1到target，内循环是对数组中的每个数。内循环计算的是以`nums[j]`为组合末尾数的方案数，因此`dp[i] += dp[i-nums[j]]`。

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target+1];
        dp[0] = 1;
        for(int i = 1; i <= target; i++) {
            for(int j = 0; j < nums.length; j++) {
                if(i >= nums[j]) dp[i] += dp[i-nums[j]];               
            }
        }
        return dp[target];
    }
}
```

### 383. 赎金信

给定一个赎金信 (`ransom`) 字符串和一个杂志(`magazine`)字符串，判断第一个字符串 `ransom` 能不能由第二个字符串 `magazines` 里面的字符构成。如果可以构成，返回 `true` ；否则返回 `false`。

(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)

**示例 1：**

```
输入：ransomNote = "a", magazine = "b"
输出：false
```

**示例 2：**

```
输入：ransomNote = "aa", magazine = "ab"
输出：false
```

**示例 3：**

```
输入：ransomNote = "aa", magazine = "aab"
输出：true
```

链接：https://leetcode-cn.com/problems/ransom-note

#### 解法：哈希表

说白了就是要验证ransomNote里的每个字符magazine里都要有，只可多不可少。

用一个数组来实现哈希表的功能

```java
class Solution {
    public boolean canConstruct(String ransomNote, String magazine) {
        if(ransomNote.length() > magazine.length()) return false;
        int[] vol = new int[26];
        for(int i = 0; i < magazine.length(); i++){
            vol[magazine.charAt(i) - 'a']++;
        }
        for(int i = 0; i < ransomNote.length(); i++) {
            if((--vol[ransomNote.charAt(i) - 'a']) < 0) return false;
        }
        return true;
    }
}
```

### 392. 判断子序列

给定字符串 **s** 和 **t** ，判断 **s** 是否为 **t** 的子序列。字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，`"ace"`是`"abcde"`的一个子序列，而`"aec"`不是）。

**进阶：**如果有大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？

**示例 1：**

```
输入：s = "abc", t = "ahbgdc"
输出：true
```

**示例 2：**

```
输入：s = "axc", t = "ahbgdc"
输出：false
```

链接：https://leetcode-cn.com/problems/is-subsequence

#### 解法一：动态规划

用布尔数组`dp[i][j]`表示`[s[0]..s[i]]`与`t[0]..t[i]`（字符串**t**中可能删除了一些字符）能否对应得上。

- 如果`dp[i-1][j-1] = true`，且`s[i]=t[j]`或者`dp[i][j-1] = true`，则`dp[i][j]=true`。
- 如果`dp[i-1][j-1] = false`，`dp[i][j]`一定是`false`。

```java
class Solution {
    public boolean isSubsequence(String s, String t) {
        int len1 = s.length(), len2 = t.length();
        if(len1 == 0) return true;
        if(len1 > len2) return false;
        char[] as = s.toCharArray();
        char[] at = t.toCharArray();
        boolean[][] dp = new boolean[len1][len2];
        for(int j = 0; j < len2; j++) {
            dp[0][j] = (as[0] == at[j]) || (j > 0 && dp[0][j-1]);
        }
        for(int i = 1; i < len1; i++) {
            for(int j = i; j < len2; j++) {
                if(dp[i-1][j-1]) dp[i][j] = (as[i] == at[j]) || dp[i][j-1];
            }
        }
        return dp[len1-1][len2-1];
    }
}
```

时间复杂度：O（mn）

还有另一种动态规划的写法。`dp[i][j]`表示的是`[s[0]..s[i]]`与`t[0]..t[i]`能按顺序匹配上的最大字符数。如果数组最后一个值不等于字符串 **s ** 的长度，结果就是false。

- 如果`s[i] == s[j]`，那么匹配上的字符可以在`dp[i-1][j-1]`基础上加一
- 如果不相等，代表对应不上，保留前一个的计算结果。

```java
class Solution {
    public boolean isSubsequence(String s, String t) {
        int len1 = s.length(), len2 = t.length();
        if(len1 == 0) return true;
        if(len2 == 0) return false;
        char[] as = s.toCharArray();
        char[] at = t.toCharArray();
        int[][] dp = new int[len1+1][len2+1];
        for(int i = 1; i <= len1; i++) {
            for(int j = 1; j <= len2; j++) {
                if(as[i-1] == at[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }else {
                    dp[i][j] = dp[i][j-1];
                }
            }
        }
        return dp[len1][len2] == len1;
    }
}
```

时间复杂度：O（mn）

#### 解法二：贪心+双指针

用两个指针分别指向字符串 **s** 和字符串 **t** 的开头，每经过一次循环`j`指针都要加一，只有当`s[i] == s[j]`的时候，`i`指针才加一。如果最后`i`指针达到了字符串 **s** 的末尾，就代表每个字符都能与 **t** 中的对应上。

```java
class Solution {
    public boolean isSubsequence(String s, String t) {
        int len1 = s.length(), len2 = t.length();
        if(len1 == 0) return true;
        if(len2 == 0) return false;
        char[] as = s.toCharArray();
        char[] at = t.toCharArray();
        int p1 = 0, p2 = 0;
        while(p1 < len1 && p2 < len2) {
            if(as[p1] == at[p2]) p1++;
            p2++;
        }
        return p1 == len1;
    }
}
```

时间复杂度：O（m）

### 404. 左叶子之和

计算给定二叉树的所有左叶子之和。

示例：

    	3
       / \
      9  20
        /  \
       15   7
    
    在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
 链接：https://leetcode-cn.com/problems/sum-of-left-leaves

#### 解法一：深度优先搜索

有好几种写的方法.

首先第一种，可以写一个辅助函数，传进去一个布尔值代表当前节点是上层的左树还是右树。只有当的确为左子节点，且当前节点的左右节点都为空的时候，才是一个左叶子。

```java
class Solution {    
    public int sumOfLeftLeaves(TreeNode root) {
        if(root == null) return 0;
        return sumOfLeftLeaves(root.left, true) + sumOfLeftLeaves(root.right, false);
    }
    public int sumOfLeftLeaves(TreeNode node, boolean isLeft) {
        if(node == null) return 0;
        if(node.left == node.right && isLeft) return node.val;

        return sumOfLeftLeaves(node.right, false) + sumOfLeftLeaves(node.left, true);
    }
}
```

第二种写法，用一个辅助函数，递归当前节点的左节点和找到其右子树中有左节点的那个节点。要用到辅助函数的原因是，我们需要事先判断根节点是否为一个孤点。

```java
class Solution {
    public int sumOfLeftLeaves(TreeNode root) {
        if(root == null || (root.left == null && root.right == null)) return 0;
        return helper(root);
    }
    private int helper(TreeNode root) {
        if(root.left == null && root.right == null) return root.val;
        int sum = 0;
        if(root.left != null) sum += helper(root.left);
        while(root.right != null) {
            root = root.right;
            if(root.left != null) {
                sum += helper(root);
                break;
            }
        }
        return sum;
    }
}
```

前面写法比较麻烦的点就在于排除根节点是孤点的情况，但是可以利用一种回退后再处理的思想。就是先递归左右节点，弹栈后再判断一个节点是否是一个左叶子。

```java
class Solution {
    public int sumOfLeftLeaves(TreeNode root) {
        if(root == null) return 0;
        int sum = 0;
        sum += sumOfLeftLeaves(root.left);
        sum += sumOfLeftLeaves(root.right);
        //判断当前节点的左边节点是否为左叶子。
        if(root.left!=null && root.left.left==null && root.left.right==null) sum += root.left.val;
        return sum;
    }
}
```

#### 解法二：广度优先搜索

层序遍历，看每个节点的左节点是否为左叶子。

```java
class Solution {
    public int sumOfLeftLeaves(TreeNode root) {
        if(root == null) return 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int sum = 0;
        while(!queue.isEmpty()) {
            root = queue.poll();
            if(root.left != null) {
                //左树不为空
                if(root.left.left == null && root.left.right == null) {
                    //左树是个左叶子
                    sum += root.left.val;
                }else {
                    queue.offer(root.left);
                }
            }
            if(root.right != null && (root.right.left != null || root.right.right != null)) {
                //右树不为空，且右树有左叶子的可能
                queue.offer(root.right);
            }
        }
        return sum;
    }
}
```

### 405. 数字转换为十六进制数

给定一个整数，编写一个算法将这个数转换为十六进制数。对于负整数，我们通常使用 [补码运算](https://baike.baidu.com/item/补码/6854613?fr=aladdin) 方法。

**注意:**

1. 十六进制中所有字母(`a-f`)都必须是小写。
2. 十六进制字符串中不能包含多余的前导零。如果要转化的数为0，那么以单个字符`'0'`来表示；对于其他情况，十六进制字符串中的第一个字符将不会是0字符。
3. 给定的数确保在32位有符号整数范围内。
4. **不能使用任何由库提供的将数字直接转换或格式化为十六进制的方法**。

**示例 1：**

```
输入:
26

输出:
"1a"
```

**示例 2：**

```
输入:
-1

输出:
"ffffffff"
```

链接：https://leetcode-cn.com/problems/convert-a-number-to-hexadecimal/

#### 解法一：常规思路

对每个数做除以及求余操作。注意要对负数做特殊处理。

```java
class Solution {
    public String toHex(int num) {
        if(num==0) return "0";
        StringBuilder sb =new StringBuilder();
        long temp = num;
        if(num<0) {
            temp=Long.valueOf(Integer.toUnsignedString(num));
        }
        while(temp!=0){
            long ys = temp%16;
            switch((int)ys){
                case 15 : sb.insert(0,'f');
                        break;
                case 14 : sb.insert(0,'e');
                        break;
                case 13 : sb.insert(0,'d');
                        break;  
                case 12 : sb.insert(0,'c');
                        break;
                case 11 : sb.insert(0,'b');
                        break;
                case 10 : sb.insert(0,'a');
                        break;
                default : sb.insert(0,ys);             
            }
            temp/=16;
        }
        return sb.toString();
    }
}
```

#### 解法二：位操作

```java
class Solution {
    public String toHex(int num) {
        if (num == 0) return "0";
        //对应数字
        char[] hex = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
        StringBuilder sb = new StringBuilder();
        while(num != 0){
            sb.append(hex[num & 0xf]);
        	// >>>：无符号右移。无论是正数还是负数，高位通通补0
            num >>>=  4;
        }
        //翻转字符串（也可以在前面的操作中不是append而是insert(0,char)
        return sb.reverse().toString();
    }
}
```

### 406. 根据身高重建队列

假设有打乱顺序的一群人站成一个队列，数组 `people` 表示队列中一些人的属性（不一定按顺序）。每个 `people[i] = [hi, ki]` 表示第 `i` 个人的身高为 `hi` ，前面 **正好** 有 `ki` 个身高大于或等于 `hi` 的人。

请你重新构造并返回输入数组 `people` 所表示的队列。返回的队列应该格式化为数组 `queue` ，其中 `queue[j] = [hj, kj]` 是队列中第 `j` 个人的属性（`queue[0]` 是排在队列前面的人）。

**示例 1：**

```
输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
解释：
编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
```

**示例 2：**

```
输入：people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
输出：[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
```

链接：https://leetcode-cn.com/problems/queue-reconstruction-by-height

#### 解法：贪心

对身高从高到低排序，按照k升序排序。

如果我们按照排完序后的顺序，依次将每个人放入队列中，那么当我们放入第 i 个人时：

第 0,⋯,i−1 个人已经在队列中被安排了位置，他们只要站在第 i 个人的前面，就会对第 i 个人产生影响，因为他们都比第 i 个人高；

而第 i+1,⋯,n−1 个人还没有被放入队列中，并且他们无论站在哪里，对第 i 个人都没有任何影响，因为他们都比第 i 个人矮。

```java
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        int length = people.length;
        if (length == 0 || people[0].length == 0) return new int[0][0];
        Arrays.sort(people, new Comparator<int[]>()
        { //按照身高降序 K升序排序
			public int compare(int[] o1, int[] o2) {
				return o1[0] == o2[0] ? o1[1] - o2[1] : o2[0] - o1[0];
			}
		});
        List<int[]> res = new ArrayList<int[]>();
        for(int i = 0; i < length; i++) {
            res.add(people[i][1], people[i]);
        }
        return res.toArray(new int[length][2]);
    }
}
```

### 409. 最长回文串

给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。在构造过程中，请注意区分大小写。比如 `"Aa"` 不能当做一个回文字符串。

**注意:**
假设字符串的长度不会超过 1010。

**示例 1:**

```
输入:
"abccccdd"

输出:
7

解释:
我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。
```

链接：https://leetcode-cn.com/problems/longest-palindrome/

#### 解法：贪心

在一个回文串中，只有最多一个字符出现了奇数次，其余的字符都出现偶数次。所以只需要判断，当前已构成的字符串长度是否是奇数，就能知道到底该再添加几个字符。

```java
class Solution {
    public int longestPalindrome(String s) {
        int[] count = new int[128];
        int length = s.length();
        //统计s里各个字符的个数
        for (int i = 0; i < length; i++) {
            char c = s.charAt(i);
            count[c]++;
        }
        int ans = 0;
        for (int v: count) {
            ans += v;
            if(ans % 2 == 0 && v % 2 != 0) ans--;
        }
        return ans;
    }
}
```

### 413. 等差数列划分

如果一个数列至少有三个元素，并且任意两个相邻元素之差相同，则称该数列为等差数列。

例如，以下数列为等差数列:

```
1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9
```

以下数列不是等差数列。

```
1, 1, 2, 5, 7
```

数组 A 包含 N 个数，且索引从0开始。数组 A 的一个子数组划分为数组 (P, Q)，P 与 Q 是整数且满足 0<=P<Q<N 。

如果满足以下条件，则称子数组(P, Q)为等差数组：

元素 A[P], A[p + 1], ..., A[Q - 1], A[Q] 是等差的。并且 P + 1 < Q 。

函数要返回数组 A 中所有为等差数组的子数组个数。

**示例:**

```
A = [1, 2, 3, 4]

返回: 3, A 中有三个子等差数组: [1, 2, 3], [2, 3, 4] 以及自身 [1, 2, 3, 4]。
```

链接：https://leetcode-cn.com/problems/arithmetic-slices

#### 解法：动态规划

用一个数组`dp[i]`存储以数字`nums[i]`为结尾的等差子数组的个数。

如果当前数字与之前两个数字构成了等差数列，即`nums[i]-nums[i-1]=nums[i-1]-nums[i-2]`，那么可以构成等差子数组的个数就在`dp[i-1]`的基础上又多了一个`dp[i] = dp[i-1]+1`。

考虑A=[3,4,5,6,7,8,9], 当前已经计算出dp[2]=1, dp[3]=2,需要求dp[4]。dp[4]=dp[3]+1 的原因是： 以A[3]=6结尾的等差数列已经有了dp[3]=2个：[3,4,5,6]和[4,5,6]。想要知道以A[4]=7结尾的等差数列个数，那么需要在已经有的dp[3]个等差数列的尾部接上一个A[4]=7, 还有一个新等差数列[5,6,7]，每次都会产生这个长度为3的等差数列。 所以才有了dp[i-1]+1。

```java
class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        int len = nums.length;
        if(len <= 2) return 0;
        int[] dp = new int[len];
        int sum = 0;
        for(int i = 2; i < len; i++) {
            if(nums[i] - nums[i-1] == nums[i-1] - nums[i-2]) {
                dp[i] = dp[i-1] + 1;
                sum += dp[i];
            }
        }
        return sum;
    }
}
```

空间上还可以优化，因为`dp[i]`只由`dp[i-1]`决定。

```java
class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        int len = nums.length;
        if(len <= 2) return 0;
        int prev = 0;
        int sum = 0;
        for(int i = 2; i < len; i++) {
            if(nums[i] - nums[i-1] == nums[i-1] - nums[i-2]) {
                prev++;
                sum += prev;
            }else {
                prev = 0;
            }
        }
        return sum;
    }
}
```

### 416. 分割等和子集

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**示例 1：**

```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
```

**示例 2：**

```
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
```

链接：https://leetcode-cn.com/problems/partition-equal-subset-sum

#### 解法：动态规划

这题本质就是一个0-1背包问题，一个背包里有一些物品，每个物品只可以取一次，问能否取出总和为XX的物品。

用布尔数组`dp[i]`表示能否分得一个总和为`i`的背包。

外层循环遍历每个物品，内层遍历质量总和（之所以要倒序，是要思考是否要利用上一轮未修改的结果，实在不清楚可以先写一遍二维数组的形式）

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for(int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if(sum%2 == 1) return false;
        sum/=2;
        boolean[] dp = new boolean[sum+1];
        dp[0] = true;
        for(int n : nums) {
            for(int j = sum; j >= n; j--) {
                if(dp[sum]) return true;
                dp[j] = dp[j] || dp[j-n];
            }
        }
        return dp[sum];
    }
}
```

或者也可以用int数组，此时`dp[i]`表示的是总和不超过`i`的能取得的最大背包质量。

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for(int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if(sum%2 == 1) return false;
        sum/=2;
        int[] dp = new int[sum+1];
        for(int i = 0; i < nums.length; i++) {
            for(int j = sum; j >= nums[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j-nums[i]] + nums[i]);
            }
        }
        return dp[sum] == sum;
    }
}
```

### 417. 太平洋大西洋水流问题

给定一个 `m x n` 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。

规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。

请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。

**提示：**

1. 输出坐标的顺序不重要
2. m 和 n 都小于150

**示例：**

```
给定下面的 5x5 矩阵:

  太平洋 ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * 大西洋

返回:

[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (上图中带括号的单元).
```

链接：https://leetcode-cn.com/problems/pacific-atlantic-water-flow

#### 解法：深度优先遍历

用两个布尔数组，分别记录能到达太平洋和大西洋的位置。从太平洋和大西洋边界开始遍历递归。

最后遍历所有坐标点，如果一个坐标点既能到太平洋也能到大西洋，就将它加入到结果数组中。

```java
class Solution {
    int m;
    int n;
    public List<List<Integer>> pacificAtlantic(int[][] matrix) {
        m = matrix.length;
        n = matrix[0].length;
        List<List<Integer>> res = new ArrayList<>();
        if(m == 0 || n == 0) return res;
        boolean[][] visP = new boolean[m][n];
        boolean[][] visA = new boolean[m][n];
        for(int i = 0; i < n; i++) {
            dfs(matrix, 0, i, visP);
            dfs(matrix, m-1, i, visA);
        }
        for(int i = 0; i < m; i++) {
            dfs(matrix, i, 0, visP);
            dfs(matrix, i, n-1, visA);
        }
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(visA[i][j] && visP[i][j]) res.add(Arrays.asList(i,j));
            }
        }
        return res;
    }
    private void dfs(int[][] matrix, int i, int j, boolean[][] visited) {
        if(visited[i][j]) return;
        visited[i][j] = true;
        //上
        if(i > 0 && matrix[i-1][j] >= matrix[i][j]) dfs(matrix, i-1, j, visited);
        //下
        if(i < m-1 && matrix[i+1][j] >= matrix[i][j]) dfs(matrix, i+1, j, visited);
        //左
        if(j > 0 && matrix[i][j-1] >= matrix[i][j]) dfs(matrix, i, j-1, visited);
        //右
        if(j < n-1 && matrix[i][j+1] >= matrix[i][j]) dfs(matrix, i, j+1, visited);
    }
}
```

有没有一个方法可以不用最后遍历一次所有的格子呢？

需要变一下记录的数组类型。`arrive[i][j]`是坐标`[i,j]`能到达的海域（太平洋sea = 1, 大西洋sea = 2）加和。

```java
class Solution {
    int[] dir = {-1, 0, 1, 0, -1};
    int[][] arrive;
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> pacificAtlantic(int[][] matrix) {        
        int m = matrix.length;
        if(m == 0) return result;
        int n = matrix[0].length;
        if(n == 0) return result;//不能在未判断m是否为0的情况下直接使用matrix[0],会造成空指针引用。
        arrive = new int[m][n];
        for(int i = 0; i < n; i++) {
            dfs(matrix, 0, i, 1);
        }
        for(int i = 0; i < m; i++) {
            dfs(matrix, i, 0, 1);
        }
        for(int i = 0; i < n; i++) {
            dfs(matrix, m - 1, i, 2);
        }
        for(int i = 0; i < m; i++) {
            dfs(matrix, i, n - 1, 2);
        }
        return result;
    }
    private void dfs(int[][] matrix, int i, int j, int sea) {
        if(arrive[i][j] >= sea) return;
        arrive[i][j] += sea;
        if(arrive[i][j] == 3) result.add(new ArrayList<Integer>(Arrays.asList(i,j)));
        for(int m = 0; m < 4; m++) {
            int x = i + dir[m], y = j + dir[m + 1];
            if(x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length) {
                if(matrix[x][y] >= matrix[i][j]) dfs(matrix, x, y, sea);        
            }
        }
    }
}
```

### 429. N叉树的层序遍历

给定一个 N 叉树，返回其节点值的*层序遍历*。（即从左到右，逐层遍历）。树的序列化输入是用层序遍历，每组子节点都由 null 值分隔（参见示例）。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

```
输入：root = [1,null,3,2,4,null,5,6]
输出：[[1],[3,2,4],[5,6]]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

```
输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
输出：[[1],[2,3,4,5],[6,7,8,9,10],[11,12,13],[14]]
```

链接：https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal

#### 解法一：广度优先搜索

```java
class Solution {
    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        result.add(Arrays.asList(root.val));
        while(!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            for(int i = 0; i < size; i++) {
                root = queue.poll();
                if(root.children == null) continue;
                for(Node node : root.children) {
                    queue.offer(node);
                    list.add(node.val);
                }
            } 
            if(list.size() != 0) result.add(list);           
        }
        return result;
    }
}
```

#### 解法二：深度优先搜索

用dfs需要知道当前节点是第几层。

```java
class Solution {
    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        dfs(result,root,0);
        return result;
    }
    private void dfs(List<List<Integer>> result, Node node, int level) {
        if(node == null) return;
        if(level >= result.size()) result.add(new ArrayList<Integer>());
        result.get(level).add(node.val);
        for(Node no: node.children) {
            dfs(result, no, level+1);
        }
    }
}
```

### 435. 无重叠区间

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

注意:

可以认为区间的终点总是大于它的起点。
区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。
**示例 1:**

```
输入: [ [1,2], [2,3], [3,4], [1,3] ]

输出: 1

解释: 移除 [1,3] 后，剩下的区间没有重叠。
```

**示例 2:**

```
输入: [ [1,2], [1,2], [1,2] ]

输出: 2

解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
```

**示例 3:**

```
输入: [ [1,2], [2,3] ]

输出: 0

解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
```

链接：https://leetcode-cn.com/problems/non-overlapping-intervals

#### 解法：贪心

先对二维数组按照起始点大小顺序进行排序，然后保留重叠区间里较小的末位点，这样能保证删除的重叠区间数量最小。

```java
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        if(intervals.length == 0) return 0;
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare (int[] a, int[] b) {
                return a[0] - b[0];
            }
        });
        int cnt = 0, prevEnd = intervals[0][1];
        for(int i = 1; i < intervals.length; i++) {
            if(intervals[i][0] < prevEnd) {
                cnt++;
                prevEnd = Math.min(intervals[i][1], prevEnd);
            }else {
                prevEnd = intervals[i][1];
            }
        }
        return cnt;
    }
}
```

时间复杂度：O（nlogn）

### 450. 删除二叉搜索树中的节点

给定一个二叉搜索树的根节点 **root** 和一个值 **key**，删除二叉搜索树中的 **key** 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

1. 首先找到需要删除的节点；
2. 如果找到了，删除它。

**说明：** 要求算法时间复杂度为 O(h)，h 为树的高度。

**示例:**

```
root = [5,3,6,2,4,null,7]
key = 3

    5
   / \
  3   6
 / \   \
2   4   7

给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。

一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。

    5
   / \
  4   6
 /     \
2       7

另一个正确答案是 [5,2,6,null,4,null,7]。

    5
   / \
  2   6
   \   \
    4   7
```

链接：https://leetcode-cn.com/problems/delete-node-in-a-bst

#### 解法一：递归

通过二叉搜索树搜寻要删除的节点，找到要删除的节点之后，把其左子树挂到右子树最左边节点上。

```java
class Solution {
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root == null) return null;
        if(root.val == key) {
            if(root.left == null) return root.right;
            if(root.right == null) return root.left;
            TreeNode node = root.right;
            while(node.left != null) node = node.left; //找最左边节点
            node.left = root.left;
            return root.right;
        } else if(root.val > key) {
            root.left = deleteNode(root.left, key);
        } else {
            root.right = deleteNode(root.right, key);
        }
        return root;
    }
}
```

#### 解法二：迭代

迭代的写法要繁琐一点，但是基本操作思路和上面是一样的。

比较特殊的是这里用到了三个变量，`fake`是一个伪头节点，`node`用于寻找要删除的节点，`prev`保存要删除的节点的父节点。

```java
class Solution {
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root == null) return null;
        TreeNode fake = new TreeNode(Integer.MAX_VALUE,root,null);
        TreeNode node = root;
        TreeNode prev = fake;
        while(node != null && node.val != key) {
            prev = node;
            if(node.val < key) {
                node = node.right;
            }else {
                node = node.left;
            }
        }
        if(node == null) return root;
        if(prev.val > key) {
            if(node.left == null) {
                prev.left = node.right;
            }else if(node.right == null) {
                prev.left = node.left;
            }else {
                TreeNode tmp = node.right;
                while(tmp.left != null) tmp = tmp.left;
                tmp.left = node.left;
                prev.left = node.right;
            }
        }else {
            if(node.left == null) {
                prev.right = node.right;
            }else if(node.right == null) {
                prev.right = node.left;
            }else {
                TreeNode tmp = node.right;
                while(tmp.left != null) tmp = tmp.left;
                tmp.left = node.left;
                prev.right = node.right;
            }            
        }
        return fake.left;
    }
}
```

### 451. 根据字符出现频率排序

给定一个字符串，请将字符串里的字符按照出现的频率降序排列。

**示例 1:**

```
输入:
"tree"

输出:
"eert"

解释:
'e'出现两次，'r'和't'都只出现一次。
因此'e'必须出现在'r'和't'之前。此外，"eetr"也是一个有效的答案。
```

**示例 2:**

```
输入:
"cccaaa"

输出:
"cccaaa"

解释:
'c'和'a'都出现三次。此外，"aaaccc"也是有效的答案。
注意"cacaca"是不正确的，因为相同的字母必须放在一起。
```

**示例 3:**

```
输入:
"Aabb"

输出:
"bbAa"

解释:
此外，"bbaA"也是一个有效的答案，但"Aabb"是不正确的。
注意'A'和'a'被认为是两种不同的字符。
```

链接：https://leetcode-cn.com/problems/sort-characters-by-frequency

#### 解法一：哈希表

用数组哈希表存储字符对应的频率，然后每添加一种字符就要重新搜索一遍寻找最大值。

```java
class Solution {
    public String frequencySort(String s) {
        int[] counter = new int[128];
        char[] res = new char[s.length()];
        for(char c : s.toCharArray()) {
            counter[c]++;
        }
        int stop = 0, p = 0, max = 0;
        while(stop < res.length) {
            for(int i = 0; i < 128; i++) {
                if(counter[i] > max) {
                    max = counter[i];
                    p = i;
                }
            }
            while(max-- > 0) {
                res[stop++] = (char)p;
            }
            counter[p] = 0;
        }
        return new String(res);
    }
}
```

时间复杂度：O（n）

#### 解法二：桶排序

依旧使用数组哈希存储字符对应的频率，然后用一个String数组，索引`i`代表字符出现频次，`String[i]`是出现该频次的所有字符。

```java
class Solution {
    public String frequencySort(String s) {
        int[] counter = new int[128];
        StringBuilder res = new StringBuilder();
        String[] record = new String[s.length()+1];
        for(char c : s.toCharArray()) {
            counter[c]++;
        }
        for(int i = 0; i < 128; i++) {
            if(counter[i] == 0) continue;
            StringBuilder sb = new StringBuilder();
            if(record[counter[i]] != null) sb.append(record[counter[i]]);
            char c = (char) i;
            for(int j = 0; j < counter[i]; j++) {
                sb.append(c);
            }
            record[counter[i]] = sb.toString();
        }
        int index = 0;
        for(int i = s.length(); i > 0 && index < s.length(); i--) {
            if(record[i] == null) continue;
            res.append(record[i]);
        }
        return new String(res);
    }
}
```

时间复杂度：O（n）

### 452. 用最少数量的箭引爆气球

在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。

一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。

给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。

**示例 1：**

```
输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球
```

**示例 2：**

```
输入：points = [[1,2],[3,4],[5,6],[7,8]]
输出：4
```

**示例 3：**

```
输入：points = [[1,2],[2,3],[3,4],[4,5]]
输出：2
```

**示例 4：**

```
输入：points = [[1,2]]
输出：1
```

**示例 5：**

```
输入：points = [[2,3],[2,3]]
输出：1
```

链接：https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons


#### 解法一：贪心

其实这题本质就是找重叠区间。先按照每个区间的起始点的大小排序，然后从前往后，看当前区间的起始点位置是否小于之前重叠区间的末尾，如果是，那么代表，可以利用同一支箭，并更新重叠区间末尾，取最小的区间末尾。如果不能，则箭数量增加，更新重叠区间末尾为当前区间的末尾值。

```java
class Solution {
    public int findMinArrowShots(int[][] points) {
        int l = points.length;
        if(l <= 1) return l;
        int nums = 1;
        Arrays.sort(points, (a, b) -> Integer.compare(a[0], b[0]));
        int prevEnd = points[0][1];        
        for(int i = 1; i < l; i++) {
            if(points[i][0] <= prevEnd) {
                prevEnd = Math.min(points[i][1], prevEnd);
            }else {
                prevEnd = points[i][1];
                nums++;
            }
        }
        return nums;
    }
}
```

也可以对每个区间的末尾点进行排序，若当前区间小于等于重叠区间的末尾，那么用同一支箭，且不用更新重叠区间的末尾值！如果大于，则更新末尾值，并对箭数加一。

```java
import java.util.Arrays;
class Solution {
    public int findMinArrowShots(int[][] points) {
        int l = points.length;
        if(l <= 1) return l;
        int nums = 1;
        Arrays.sort(points, (a, b) -> Integer.compare(a[1], b[1]));
        int prev = points[0][1];
        
        for(int i = 1; i < l; i++) {
            if(points[i][0] > prev) {
                nums++;
                prev = points[i][1];
            }
        }
        return nums;
    }
}
```

### 454. 四数相加 II

给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 `(i, j, k, l)` ，使得 `A[i] + B[j] + C[k] + D[l] = 0`。为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。所有整数的范围在 -228 到 228 - 1 之间，最终结果不会超过 231 - 1 。

**例如:**

```
输入:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

输出:
2

解释:
两个元组如下:

1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
```


链接：https://leetcode-cn.com/problems/4sum-ii

#### 解法：分组+哈希表

将四个数组分成两部分，A和B一组，C和D一组。将A和B的所有和及出现次数存入哈希表，然后对C和D求和，看是否在哈希表中有对应的可以与其构成和为0的值存在。

```java
class Solution {
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        HashMap<Integer,Integer> hm = new HashMap<>();
        int count = 0;
        for(int a : nums1) {
            for(int b : nums2) {
                hm.compute(a + b, (k, v) -> v == null ? 1 : v + 1);
            }
        }
        for(int c : nums3) {
            for(int d : nums4) {
                count+=hm.getOrDefault(-(c+d),0);
            }
        }
        return count;
    }
}
```

时间复杂度：O（n^2^）

### 455. 分发饼干

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。对每个孩子 `i`，都有一个胃口值 `g[i]`，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 `j`，都有一个尺寸 `s[j]` 。如果 `s[j] >= g[i]`，我们可以将这个饼干 `j` 分配给孩子 `i` ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

**示例 1:**

```
输入: g = [1,2,3], s = [1,1]
输出: 1
解释: 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。
```

**示例 2:**

```
输入: g = [1,2], s = [1,2,3]
输出: 2
解释: 
你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出2.
```

链接：https://leetcode-cn.com/problems/assign-cookies

#### 解法：贪心

尽量选能满足孩子胃口的最小的饼干。

```java
class Solution {
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int index = 0;
        int res = 0;
        for(int apetite : g) {
            while(index < s.length && s[index] < apetite) {
                index++;
            }
            if(index >= s.length) break;
            res++;
            index++;          
        }
        return res;
    }
}
```

### 474. 一和零

给你一个二进制字符串数组 `strs` 和两个整数 `m` 和 `n` 。请你找出并返回 `strs` 的最大子集的大小，该子集中 **最多** 有 `m` 个 `0` 和 `n` 个 `1` 。如果 `x` 的所有元素也是 `y` 的元素，集合 `x` 是集合 `y` 的 **子集** 。

**示例 1：**

```
输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
输出：4
解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。
```

**示例 2：**

```
输入：strs = ["10", "0", "1"], m = 1, n = 1
输出：2
解释：最大的子集是 {"0", "1"} ，所以答案是 2 。
```

链接：https://leetcode-cn.com/problems/ones-and-zeroes

#### 解法：动态规划

一开始想用回溯法写，但是回溯法估计会超时，因为必定会重复计算。

这题的本质就是01背包问题。用数组`dp[i][j]`表示构成不超过`i`个`0`和`j`个`1`的最大子集大小。

```java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int len = strs.length;
        int[][] dp = new int[m+1][n+1];
        
        for(int i = 0; i < len; i++) {
            int zeros = 0;
            int ones = 0;
            //计算0和1的个数
            for(int j = 0; j < strs[i].length(); j++) {
                if(strs[i].charAt(j) == '0') {
                    zeros++;
                }else {
                    ones++;
                }
            }
            //遍历顺序一定要从后往前。
            for(int s = m; s >= zeros; s--) {
                for(int t = n; t >= ones; t--) {
                    dp[s][t] = Math.max(dp[s-zeros][t-ones] + 1, dp[s][t]);
                }
            }
        }
        return dp[m][n];
    }
}
```

### 494. 目标和

给你一个整数数组 `nums` 和一个整数 `target` 。向数组中的每个整数前添加 `'+'` 或 `'-'` ，然后串联起所有整数，可以构造一个 **表达式** ：例如，`nums = [2, 1]` ，可以在 `2` 之前添加 `'+'` ，在 `1` 之前添加 `'-'` ，然后串联起来得到表达式 `"+2-1"`。返回可以通过上述方法构造的、运算结果等于 `target` 的不同 **表达式** 的数目。

**示例 1：**

```
输入：nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有 5 种方法让最终目标和为 3 。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
```

**示例 2：**

```
输入：nums = [1], target = 1
输出：1
```

链接：https://leetcode-cn.com/problems/target-sum

#### 解法一：回溯法

```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        return findTargetSumWays(nums, 0, target);
    }
    public int findTargetSumWays(int[] nums, int start, int target) {
        if(start == nums.length) {
            if(target == 0) return 1;
            return 0;
        }
        return findTargetSumWays(nums, start+1, target+nums[start]) + findTargetSumWays(nums, start+1, target-nums[start]);
    }
}
```

时间复杂度：O（2^n^）

#### 解法二：动态规划

这题就是在问，该数组能否分成两部分`N`和`P`使得`P-N=target`，这样的组合有多少种。又因为数组的数值总和`N+P=sum`是可以计算出来的，因此也可以求得`P`的大小。所以就可以转化为求为 "和为`P`的子集种类"。如此便转化为了01背包问题。

```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for(int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if((sum+target)%2 == 1 || target > sum) return 0;
        int t = Math.min((sum+target)/2,(sum-target)/2);        
        int[] dp = new int[t + 1];
        dp[0] = 1;
        for(int i = 0; i < nums.length; i++) {
            for(int j = t; j >= nums[i]; j--) {
                dp[j] += dp[j-nums[i]];
            }
        }
        return dp[t];
    }
}
```

### 501. 二叉搜索树中的众数

给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。

假定 BST 有如下定义：

- 结点左子树中所含结点的值小于等于当前结点的值
- 结点右子树中所含结点的值大于等于当前结点的值
- 左子树和右子树都是二叉搜索树
  例如：

```
给定 BST [1,null,2,2],

   1
    \
     2
    /
   2
返回[2].
```

提示：如果众数超过1个，不需考虑输出顺序

进阶：你可以不使用额外的空间吗？（假设由递归产生的隐式调用栈的开销不被计算在内）
链接：https://leetcode-cn.com/problems/find-mode-in-binary-search-tree

#### 解法一：暴力

最暴力的想法就是遍历整棵树构成一个有序数组，然后找每个数出现的频次，取出众数。

这里构成有序数组的时候可以采取中序遍历，利用二叉搜索树的特性。

```java
class Solution {
    List<Integer> tree = new ArrayList<>();
    public int[] findMode(TreeNode root) {
        BST(root);
        int max = 1;
        int count = 1;
        int prev = tree.get(0);
        List<Integer> res = new ArrayList<>();
        res.add(prev);
        //--------------------------
        for(int i = 1; i < tree.size(); i++) {
            if(tree.get(i) == prev) {
                count++;
                if(count == max) {
                    res.add(tree.get(i));
                }else if(count > max) {
                    res.clear();
                    res.add(tree.get(i));
                    max = count;
                }
            }else {
                prev = tree.get(i);
                count = 1;
            }
        }
        //--------------------------
        int[] resa = new int[res.size()];
        for(int i = 0; i < res.size(); i++) {
            resa[i] = res.get(i);
        }
        return resa;
    }
    private void BST(TreeNode root) {
        if(root == null) {
            return;
        }
        BST(root.left);
        tree.add(root.val);
        BST(root.right);
    }
}
```

#### 解法二：暴力优化

其实可以想见上面的代码里做了多余的操作，在中序遍历这棵树的过程中，就是从小到大的按序遍历，和上面代码中的画杠区间是一样的顺序，那么在画杠区域中做的操作应该也可以直接在递归中做完。

保存全局遍量prev，count，max。

```java
class Solution {
    List<Integer> res = new ArrayList<>();
    TreeNode prev = null;
    int max = 0;
    int count = 0;
    public int[] findMode(TreeNode root) {
        BST(root);
        int l = res.size();
        int[] ares = new int[l];
        for(int i = 0; i < l; i++) {
            ares[i] = res.get(i);
        }
        return ares;
    }
    private void BST(TreeNode root) {
        if(root == null) {
            return;
        }

        BST(root.left);
//---------------------------------------------------
        if(prev == null || prev.val != root.val) {
            prev = root;
            count = 1;
        }else if(prev.val == root.val) {
            count++;
        }

        if(count > max) {            
            res.clear();
            max = count;
            res.add(prev.val);
        }else if(count == max) {
            res.add(prev.val);
        }
//----------------------------------------------------
        BST(root.right);
    }
}
```

#### 解法三：Morris中序遍历

用递归的方法做中序遍历，且不用额外空间：通过修改树来完成。将当前节点挂到左子树的最右边节点下面。

```java
class Solution {
    int base, count, maxCount;
    List<Integer> answer = new ArrayList<Integer>();

    public int[] findMode(TreeNode root) {
        TreeNode cur = root, pre = null;
        while (cur != null) {
            if (cur.left == null) {
                update(cur.val);
                cur = cur.right;
                continue;
            }
            pre = cur.left;
            //修改树的时候并没有断开原先的连接，所以要判断pre.right != cur
            while (pre.right != null && pre.right != cur) {
                pre = pre.right;
            }
            if (pre.right == null) {
                pre.right = cur;
                cur = cur.left;
            } else {
                pre.right = null;
                update(cur.val);
                cur = cur.right;
            }
        }
        int[] mode = new int[answer.size()];
        for (int i = 0; i < answer.size(); ++i) {
            mode[i] = answer.get(i);
        }
        return mode;
    }

    public void update(int x) {
        if (x == base) {
            ++count;
        } else {
            count = 1;
            base = x;
        }
        if (count == maxCount) {
            answer.add(base);
        }
        if (count > maxCount) {
            maxCount = count;
            answer.clear();
            answer.add(base);
        }
    }
}
```

### 504. 七进制数

给定一个整数，将其转化为7进制，并以字符串形式输出。

**示例 1:**

```
输入: 100
输出: "202"
```

**示例 2:**

```
输入: -7
输出: "-10"
```

注意: 输入范围是 [-1e7, 1e7] 。
链接：https://leetcode-cn.com/problems/base-7

这题真没什么好说的了。

```java
class Solution {
    public String convertToBase7(int num) {
        if(num == 0) return "0";
        StringBuilder sb = new StringBuilder();
        boolean isPositive = true;
        if(num < 0) {
            sb.append("-");
            num = Math.abs(num);
            isPositive = false;
        }
        while(num > 0) {
            if(!isPositive){
                sb.insert(1,num%7);
            }else {
                sb.insert(0,num%7);
            }
            num /= 7;
        }
        return sb.toString();
    }
}
```

### 509. 斐波那契数

**斐波那契数**，通常用 `F(n)` 表示，形成的序列称为 **斐波那契数列** 。数列由 `0` 和 `1` 开始，后面的每一项数字都是前面两项数字的和。也就是：

```
F(0) = 0，F(1) = 1
F(n) = F(n - 1) + F(n - 2)，其中 n > 1
```

给你 `n` ，请计算 `F(n)` 。

**示例 1：**

```
输入：2
输出：1
解释：F(2) = F(1) + F(0) = 1 + 0 = 1
```

**示例 2：**

```
输入：3
输出：2
解释：F(3) = F(2) + F(1) = 1 + 1 = 2
```

**示例 3：**

```
输入：4
输出：3
解释：F(4) = F(3) + F(2) = 2 + 1 = 3
```

链接：https://leetcode-cn.com/problems/fibonacci-number

####  解法：动态规划

经典的的动态规划题目了。应该是闭着眼都能背出来的题。

```java
class Solution {
    public int fib(int n) {
        if(n <= 1) return n;
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = 1;
        for(int i = 2; i <= n; i++) {
            dp[i] = dp[i-1]+dp[i-2];
        }
        return dp[n];
    }
}
```

也可以不用额外空间。

```java
class Solution {
    public int fib(int n) {
        if(n <= 1) return n;
        int nminus2 = 0, nminus1 = 1;
        while(n >= 2) {
            int tmp = nminus2;
            nminus2 = nminus1;
            nminus1 += tmp;
            n--;
        }
        return nminus1;
    }
}
```

### 513. 找树左下角的值

给定一个二叉树，在树的最后一行找到最左边的值。

**示例 1:**

    输入:
        2
       / \
      1   3
    
    输出:
    1
  **示例 2:**

    输入: 
       1
       / \
      2   3
     /   / \
    4   5   6
       /
      7
    输出:
    7

**注意:** 您可以假设树（即给定的根节点）不为 **NULL**。
链接：https://leetcode-cn.com/problems/find-bottom-left-tree-value

#### 解法一：层序遍历

用一个队列保存每一层的节点，记录每层第一个节点的值。

```java
class Solution {
    public int findBottomLeftValue(TreeNode root) {
    	int res = 0;
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        que.offer(root);
        while(!que.isEmpty()) {
            int size = que.size();    
            TreeNode leftFirst = null;        
            for(int i = 0; i < size; i++) {
                TreeNode node = que.poll();
                if(i == 0) res = node.val;
                if(node.left != null) que.offer(node.left);
                if(node.right != null) que.offer(node.right);
            }
        }
        return res;
    }
}
```

#### 解法二：递归

递归需要用到记录层数的参数。

```java
class Solution {
    int res = 0;
    int level = -1;
    public int findBottomLeftValue(TreeNode root) {
        dfs(root, 0);
        return res;
    }
    private void dfs(TreeNode root, int depth) {
        if(root == null) return;
        if(depth > level) {
            level = depth;
            res = root.val;
        }
        dfs(root.left, depth+1);
        dfs(root.right, depth+1);
    }
}
```

### 515. 在每个树行中找最大值

您需要在二叉树的每一行中找到最大的值。

示例：

输入: 

          1
         / \
        3   2
       / \   \  
      5   3   9 

输出: [1, 3, 9]
链接：https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row

#### 解法一：递归

这题用前序中序还是后序都没什么关系。

```java
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> largestValues(TreeNode root) {
        if(root == null) return res;
        dfs(0,root);
        return res;
    }
    private void dfs(int level, TreeNode node) {
        if(node == null) return;
        if(level >= res.size()) {
            res.add(node.val);
        }else {
            if(node.val > res.get(level)) res.set(level, node.val);
        }
        dfs(level+1,node.left);
        dfs(level+1,node.right);
    }
}
```

#### 解法二：迭代

```java
class Solution {    
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if(root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()) {
            int size = queue.size();
            int min = Integer.MIN_VALUE;
            for(int i = 0; i < size; i++) {
                root = queue.poll();
                min = Math.max(min, root.val);
                if(root.left != null) queue.offer(root.left);
                if(root.right != null) queue.offer(root.right);
            }
            res.add(min);
        }
        return res;
    }
}
```

### 516. 最长回文子序列

给定一个字符串 `s` ，找到其中最长的回文子序列，并返回该序列的长度。可以假设 `s` 的最大长度为 `1000` 。

**示例 1:**

```
输入:

"bbbab"
输出:

4
一个可能的最长回文子序列为 "bbbb"。
```

**示例 2:**

```
输入:

"cbbd"
输出:

2
一个可能的最长回文子序列为 "bb"。
```

链接：https://leetcode-cn.com/problems/longest-palindromic-subsequence

#### 解法：动态规划

用数组`dp[i][j]`保存从`i`到`j`的最长回文子串长度。如果`s[i]`和`s[j]`相等，`dp[i][j] = dp[i+1][j-1]+2`，如果不相等，`dp[i][j]`可能是等于`i-1`到`j`之间的最长子串长度，也可能是`i`到`j-1`之间的。

```java
class Solution {
    public int longestPalindromeSubseq(String s) {
        int len = s.length();
        if(len == 1) return 1;
        char[] chars = s.toCharArray();
        int[][] dp = new int[len][len];
        for(int i = 0; i < len; i++) dp[i][i] = 1;
        for(int i = len-2; i >= 0; i--) {
            //从后往前遍历是因为我们最终要求的是dp[0][len-1]，这个长度取决于(0,len-1)之间的最长子串长度，必须要先求出里面的，才能得到外部的
            for(int j = i+1; j < len; j++) {
                if(chars[i] == chars[j]) {
                    dp[i][j] = dp[i+1][j-1] + 2;
                }else {
                    dp[i][j] = Math.max(dp[i][j-1], dp[i+1][j]);
                }
            }
        }
        return dp[0][len-1];
    }
}
```

空间还可以再进一步优化。

```java
class Solution {
    public int longestPalindromeSubseq(String s) {
        int len = s.length();
        if(len == 1) return 1;
        char[] chars = s.toCharArray();
        int[] dp = new int[len];
        for(int i = 0; i < len; i++) dp[i] = 1;
        for(int i = len-2; i >= 0; i--) {
            int prev = 0;//保存dp[i+1][j-1]
            for(int j = i+1; j < len; j++) {
                //当前的dp[j]就是二维数组中的dp[i+1][j]
                //也是下一轮循环中的dp[i+1][j-1]
                int tmp = dp[j];
                if(chars[i] == chars[j]) {
                    dp[j] = prev + 2;
                }else {
                    dp[j] = Math.max(dp[j-1], dp[j]);
                }
                prev = tmp;
            }
        }
        return dp[len-1];
    }
}
```

### 518. 零钱兑换 II

给你一个整数数组 `coins` 表示不同面额的硬币，另给一个整数 `amount` 表示总金额。请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 `0` 。假设每一种面额的硬币有无限个。 题目数据保证结果符合 32 位带符号整数。

**示例 1：**

```
输入：amount = 5, coins = [1, 2, 5]
输出：4
解释：有四种方式可以凑成总金额：
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
```

**示例 2：**

```
输入：amount = 3, coins = [2]
输出：0
解释：只用面额 2 的硬币不能凑成总金额 3 。
```

**示例 3：**

```
输入：amount = 10, coins = [10] 
输出：1
```

链接：https://leetcode-cn.com/problems/coin-change-2

#### 解法：动态规划

这是个完全背包问题。用数组`dp[i][j]`表示前`i`件物品构成总和`j`有几种方式。`dp[i][j] = dp[i-1][j] + dp[i][j-nums[i]]`

可以只用一维数组。

```java
class Solution {
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount+1];
        dp[0] = 1;
        for(int coin : coins){
            for(int i = coin; i <= amount; i++){
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }
}
```

### 524. 通过删除字母匹配到字典里的最长单词

给你一个字符串 `s` 和一个字符串数组 `dictionary` 作为字典，找出并返回字典中最长的字符串，该字符串可以通过删除 `s` 中的某些字符得到。如果答案不止一个，返回长度最长且字典序最小的字符串。如果答案不存在，则返回空字符串。

**示例 1：**

```
输入：s = "abpcplea", dictionary = ["ale","apple","monkey","plea"]
输出："apple"
```

**示例 2：**

```
输入：s = "abpcplea", dictionary = ["a","b","c"]
输出："a"
```

链接：https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting

#### 解法：双指针

```java
class Solution {
    
    public String findLongestWord(String s, List<String> d) {
        String result = "";
        for (String t : d) {
            if (isSubsequence(t, s)) {
                // 获取长度最长且字典顺序最小的字符串
                if (result.length() < t.length() || (result.length() == t.length() && result.compareTo(t) > 0)) {
                    result = t;
                }
            }
        }
        return result;
    }

    // 判断 t 是否为 s 的子序列，双指针
    public boolean isSubsequence(String t, String s) {
        int i = 0, j = 0;
        while (i < s.length() && j < t.length()) {
            if (s.charAt(i) == t.charAt(j)) {
                j++;
            }
            i++;
        }
        return j == t.length();
    }
}
```

### 530. 二叉搜索树的最小绝对差

给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。

**示例：**

```
输入：

   1
    \
     3
    /
   2

输出：
1

解释：
最小绝对差为 1，其中 2 和 1 的差的绝对值为 1（或者 2 和 3）。
```

链接：https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst

#### 解法：中序遍历

这一题千万不要去找左右子树再返回求最小绝对值差。被坑过几次了。以后看到二叉搜索树首先要想到中序的特性。

一个二叉搜索树它的中序就是一个有序的递增数组，要找最小的绝对值差，就是要找两个相邻的最小差。

通过中序遍历，求当前节点与上一个节点值的差，保留得最小值。

```java
class Solution {    
    int min_res = Integer.MAX_VALUE;
    int prev = -1;//利用所有节点值都非负这一点

    public int getMinimumDifference(TreeNode root) { //中序遍历
        if(root == null) return min_res;
        getMinimumDifference(root.left);   //左

        if(prev >= 0) {
            int dif = root.val-prev;
            if(now < min_res) min_res = dif;   //中
        }        
        prev = root.val;

        return getMinimumDifference(root.right);//右
    }
}
```

### 538. 把二叉搜索树转换为累加树

给出二叉 **搜索** 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 `node` 的新值等于原树中大于或等于 `node.val` 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

- 节点的左子树仅包含键 **小于** 节点键的节点。
- 节点的右子树仅包含键 **大于** 节点键的节点。
- 左右子树也必须是二叉搜索树。

**示例 1：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/05/03/tree.png)

```
输入：[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
```

**示例 2：**

```
输入：root = [0,null,1]
输出：[1,null,1]
```

**示例 3：**

```
输入：root = [1,0,2]
输出：[3,3,2]
```

**示例 4：**

```
输入：root = [3,2,4,1]
输出：[7,9,4,10]
```

链接：https://leetcode-cn.com/problems/convert-bst-to-greater-tree

#### 解法一：递归 反中序遍历

因为二叉树的中序遍历就是一个递增数组，将数组反过来变成一个递减数组，然后求前缀和就是题目要求的。

要按照递减的顺序，反中序遍历即可，即按照 右->中->左 的顺序。

```java
class Solution {
    public TreeNode convertBST(TreeNode root) {
        if(root == null) return root;
        convertBST(root.right);   
        prev += root.val;
        root.val = prev;
        convertBST(root.left);
        return root;   
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

#### 解法二：迭代

用一个栈来模拟递归

```java
class Solution {
    public TreeNode convertBST(TreeNode root) {
        if(root == null) return root;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode node = root;
        TreeNode prev = null;
        int sum = 0;
        while(!stack.isEmpty()) {
            //如果prev == node代表在回退
            while(prev != node && node.right != null) {
                node = node.right;
                stack.push(node);
            }
            node = stack.pop();
            //用一个遍历保存之前处理的节点
            prev = node;
            sum += node.val;
            node.val = sum;
            if(node.left != null) {
                node = node.left;
                stack.push(node);
            }
        }
        return root;   
    }
}
```

时间复杂度：O（n）

空间复杂度：O（n）

#### 解法三：Morris遍历

如何不利用额外的空间完成中序遍历？

Morris遍历过程：

> **记当前节点为cur。**
>
> 1. 如果cur无左孩子，cur向右移动（cur=cur.right）
>
> 2. 如果cur有左孩子，找到cur左子树上最右的节点，记为mostright
>
> 3. 1. 如果mostright的right指针指向空，让其指向cur，cur向左移动（cur=cur.left）
>    2. 如果mostright的right指针指向cur，让其指向空，cur向右移动（cur=cur.right）

因为这里要的是反中序，所以我们不是把cur挂到左子树上最右节点，而是右子树上最左节点。

```java
class Solution {
    public TreeNode convertBST(TreeNode root) {
        TreeNode cur = root;
        TreeNode mostLeft = null;
        int sum = 0;
        while(cur != null) {
            //如果右子树不为空，要找到右子树上最左边的节点mostLeft，将当前节点cur挂到mostLeft的左节点上
            if(cur.right != null) {
                mostLeft = cur.right;
                while(mostLeft.left != null && mostLeft.left != cur) {
                    mostLeft = mostLeft.left;
                }
                if(mostLeft.left == null) {
                    mostLeft.left = cur;
                    cur = cur.right;
                }else {
                    //mostLeft.left = cur，代表是已经被我们修改过的节点，说明cur的右子树已经都遍历过了                    
                    mostLeft.left = null;//要还原树结构
                    sum += cur.val;
                    cur.val = sum;
                    cur = cur.left;
                }
            }else {
                sum += cur.val;
                cur.val = sum;
                cur = cur.left;
            }
        }
        return root;   
    }
}
```

时间复杂度：O（n）

空间复杂度：O（1）

### 540. 有序数组中的单一元素

给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。

**示例 1:**

```
输入: [1,1,2,3,3,4,4,8,8]
输出: 2
```

**示例 2:**

```
输入: [3,3,7,7,10,11,11]
输出: 10
```

**注意:** 您的方案应该在 O(log n)时间复杂度和 O(1)空间复杂度中运行。
链接：https://leetcode-cn.com/problems/single-element-in-a-sorted-array

#### 解法一：暴力法

不满足题目的时间复杂度要求，但是这里有一个可以注意的地方，就是如果循环结束都没有返回某个值的话，代表数组末尾值就是我们要找的那个

```java
class Solution {
    public int singleNonDuplicate(int[] nums) {
        if(nums.length == 1) return nums[0];
        int cnt = 0;
        int prev = 0;
        for(int i = 0; i < nums.length; i++) {
            if(cnt == 0) {
                prev = nums[i];
                cnt++;
            }else {
                if(nums[i] != prev) return prev;
                cnt = 0;
            }
        }
        return prev;
    }
}
```

#### 解法二：位运算

不满足题目的时间复杂度要求，但是对于一个不有序的数组来说，位运算是最佳做法。

利用异或运算的特性

```java
class Solution {
    public int singleNonDuplicate(int[] nums) {
        if(nums.length == 1) return nums[0];
        int res = nums[0];
        for(int i = 1; i < nums.length; i++) {
            res ^= nums[i];
        }
        return res;
    }
}
```



#### 解法三：二分搜索

因为只有一个数会出现一次，其余都是出现两次，那么存在这个单个数的数组一定长度为奇数。可以通过二分法，对左右两边长度为奇数的数组进行下一步搜索

![](https://pic.leetcode-cn.com/2af08845f26dd0f300ffa587fafd4e584461a6d2d710a89f8997b8cb0f9de9b6-file_1576478245272)

```java
class Solution {
    public int singleNonDuplicate(int[] nums) {
        if(nums.length == 1) return nums[0];
        int lo = 0, hi = nums.length - 1;
        while(lo < hi) {
            int mid = lo + (hi - lo)/2;
            if(nums[mid] != nums[mid-1] && nums[mid] != nums[mid+1]) return nums[mid];
            //找到长度为奇数的数组区间
            if(nums[mid] == nums[mid-1]) {
                if((mid - 1 - lo)%2 != 0) {
                    hi = mid - 2;
                }else {
                    lo = mid + 1;
                }
            }else {
                if((hi - mid - 1)%2 != 0) {
                    lo = mid + 2;
                }else {
                    hi = mid - 1;
                }
            }
        }
        return nums[lo];
    }
}
```

时间复杂度：O（nlogn）

空间复杂度：O（1）

### 541. 反转字符串 II

给定一个字符串 `s` 和一个整数 `k`，你需要对从字符串开头算起的每隔 `2k` 个字符的前 `k` 个字符进行反转。

- 如果剩余字符少于 `k` 个，则将剩余字符全部反转。
- 如果剩余字符小于 `2k` 但大于或等于 `k` 个，则反转前 `k` 个字符，其余字符保持原样。

**示例:**

```
输入: s = "abcdefg", k = 2
输出: "bacdfeg"
```

 链接：https://leetcode-cn.com/problems/reverse-string-ii/

#### 解法：暴力

这题就是直接翻译题意了。

```java
class Solution {
    public String reverseStr(String s, int k) {
        char[] c = s.toCharArray();
        for(int i = 0; i < c.length; i += 2*k) {
            reverse(c, i, i+k-1);
        }
        return String.valueOf(c);
    }
    private void reverse(char[] c, int start, int end) {
        if(start >= c.length) return;
        end = Math.min(end, c.length-1);
        while(start < end) {
            swap(c, start++, end--);
        }
    }
    private void swap(char[] c, int i, int j){
        char tmp = c[i];
        c[i] = c[j];
        c[j] = tmp;
    }
}
```

### 547.  省份数量

有 `n` 个城市，其中一些彼此相连，另一些没有相连。如果城市 `a` 与城市 `b` 直接相连，且城市 `b` 与城市 `c` 直接相连，那么城市 `a` 与城市 `c` 间接相连。**省份** 是一组直接或间接相连的城市，组内不含其他没有相连的城市。给你一个 `n x n` 的矩阵 `isConnected` ，其中 `isConnected[i][j] = 1` 表示第 `i` 个城市和第 `j` 个城市直接相连，而 `isConnected[i][j] = 0` 表示二者不直接相连。返回矩阵中 **省份** 的数量。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/12/24/graph1.jpg)

```
输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
输出：2
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2020/12/24/graph2.jpg)

```
输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]]
输出：3
```

链接：https://leetcode-cn.com/problems/number-of-provinces

#### 解法一：深度优先遍历

每搜索到一个相连城市，就去搜索与该城市相连的其他城市。

通过直接改变`isConnected`数组里的值来记录是否遍历过这个城市。

```java
class Solution {
    public int findCircleNum(int[][] isConnected) {
        int cnt = 0;
        for(int i = 0; i < isConnected.length; i++) {
            for(int j = 0; j < isConnected.length; j++) {
                if(isConnected[i][j] == 1) {
                    cnt++;
                    dfs(isConnected, i, j);
                }
            }
        }
        return cnt;
    }
    private void dfs(int[][] isConnected, int i, int j) {
        isConnected[i][j] = 0;
        isConnected[j][i] = 0;
        for(int m = 0; m < isConnected.length; m++) {
            if(m == j) {
                isConnected[j][j] = 0;
                continue;
            }
            if(isConnected[j][m] == 1) dfs(isConnected, j, m);
        }
    }
}
```

#### 解法二：广度优先遍历

找到与一个城市相连的所有其他城市，再找到与这些城市相连的所有城市，直到所有的都已经被查找过。

```java
class Solution {
    public int findCircleNum(int[][] isConnected) {
        int cnt = 0;
        Queue<Integer> que = new LinkedList<>();
        boolean[] visited = new boolean[isConnected.length];
        for(int i = 0; i < isConnected.length; i++) {
            if(visited[i]) continue;
            que.offer(i);
            while(!que.isEmpty()) {
                int cur = que.poll();
                visited[cur] = true;
                for(int j = 0; j < isConnected.length; j++) {
                    if(visited[j]) continue;
                    if(isConnected[cur][j] == 1) que.offer(j);
                }
            }
            cnt++;
        }
        return cnt;
    }
}
```

### 559. N叉树的最大深度

给定一个 N 叉树，找到其最大深度。

最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。

N 叉树输入按层序遍历序列化表示，每组子节点由空值分隔（请参见示例）。

**示例 1：**

![](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

```
输入：root = [1,null,3,2,4,null,5,6]
输出：3
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

```
输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
输出：5
```

链接：https://leetcode-cn.com/problems/maximum-depth-of-n-ary-tree

#### 解法一：递归 深度优先搜索

从一个节点出发，选定一个孩子，继续选定孩子的孩子一路递归。

```java
class Solution {
    public int maxDepth(Node root) {
        return dfs(root, 0);
    }
    public int dfs(Node node, int level) {
        if(node == null) return level;
        int max = level + 1;
        for(Node n : node.children) {
            max = Math.max(dfs(n, level+1), max);
        }
        return max;
    }
}
```

#### 解法二：迭代 广度优先搜索

```java
class Solution {
    public int maxDepth(Node root) {
        if(root == null) return 0;
        Queue<Node> que = new LinkedList<>();
        int level = 0;
        que.offer(root);
        while(!que.isEmpty()) {
            level++;
            int size = que.size();
            for(int i = 0; i < size; i++) {
                root = que.poll();
                for(Node node : root.children) {
                    que.offer(node);
                }
            }
        }
        return level;
    }
}
```

### 572. 另一个树的子树

给定两个非空二叉树 **s** 和 **t**，检验 **s** 中是否包含和 **t** 具有相同结构和节点值的子树。**s** 的一个子树包括 **s** 的一个节点和这个节点的所有子孙。**s** 也可以看做它自身的一棵子树。

**示例 1:**
给定的树 s:

         3
        / \
       4   5
      / \
     1   2
   给定的树 t：   

```
   4 
  / \
 1   2
```

返回 true，因为 t 与 s 的一个子树拥有相同的结构和节点值。

**示例 2:**
给定的树 s：

         3
        / \
       4   5
      / \
     1   2
        /
       0
  给定的树 t：

```
   4
  / \
 1   2
```

返回 false。
链接：https://leetcode-cn.com/problems/subtree-of-another-tree

#### 解法一：深度优先搜索暴力匹配

```java
class Solution {
    public boolean isSubtree(TreeNode s, TreeNode t) {   
        return isSubtree(s, t, true) || isSubtree(s, t, false);      
    }
    public boolean isSubtree(TreeNode s, TreeNode t, boolean startSearch) {   
        if(s == null && t == null) return true;
        if(s == null || t == null) return false;
        if(startSearch) {
            if(s.val != t.val) return false;
            return isSubtree(s.left, t.left, true) && isSubtree(s.right, t.right, true);
        } else {
            if(s.val == t.val) {
                //以当前节点为子树的根节点开始搜索，如果不相同的话，还要继续对下面的节点进行搜索
                boolean flag = isSubtree(s.left, t.left, true) && isSubtree(s.right, t.right, true);
                if(flag) return true;
            }
            return isSubtree(s.left, t, false) || isSubtree(s.right, t, false);
        }     
    }
}
```

另一种写法，比较简洁

```java
class Solution {
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if(s == null && t == null) return true;
        if(s == null || t == null) return false;        
        return isSubtree(s.left, t) || isSubtree(s.right, t) || isSame(s,t);            
    }
    public boolean isSame(TreeNode s, TreeNode t) {
        if(s == null && t == null) return true;
        if(s == null || t == null) return false;
        if(s.val == t.val) return isSame(s.left,t.left) && isSame(s.right,t.right);
        return false;             
    }
}
```

时间复杂度：O（s*t）

#### 解法二： KMP

将两棵树的前序遍历序列记录下来，匹配`s`串里有没有`t`串（这部分可以看28题）。

有一个问题就是，对于上面示例2，观察两个前序遍历序列[3,4,1,2,0,5] 和 [4,1,2] 虽然有匹配到，但是树t并不是s 的子树。

为了解决这个问题要引入`lNULL`和`rNULL`，这样得到的前序遍历序列就是[3,4,1, lNULL, rNULL, 2,0,rNULL,5]，就匹配不到[4,1,2]了。

为了方便，`lNULL`取最小值，`rNULL`取最大值。要注意Integer比较的时候最好用`equals`方法，因为**默认IntegerCache.low 是-127，Integer.high是128，如果在这个区间内，他就会把变量i当做一个变量，放到内存中；但如果不在这个范围内，就会去new一个Integer对象**

```java
 class Solution {
    int lNULL = Integer.MIN_VALUE, rNULL = Integer.MAX_VALUE;
    public boolean isSubtree(TreeNode s, TreeNode t) {
        List<Integer> sOrder = new ArrayList<>();
        List<Integer> tOrder = new ArrayList<>();
        sOrder.add(s.val);
        tOrder.add(t.val);
        getOrder(s, sOrder);
        getOrder(t, tOrder);
        return kmp(sOrder, tOrder);
    }
    private void getOrder(TreeNode root, List<Integer> list) {
        if(root == null) return;
        if(root.left == null) {
            list.add(lNULL);
        }else {
            list.add(root.left.val);
            getOrder(root.left, list);
        }
        if(root.right == null) {
            list.add(rNULL);
        }else {
            list.add(root.right.val);
            getOrder(root.right, list);
        }
    }
    private boolean kmp(List<Integer> list1, List<Integer> list2) {
        if(list1.size() < list2.size()) return false;
        int[] next = new int[list2.size()];
        next[0] = -1;
        int k = -1, i = 0;
        while(i < list2.size()-1) {
            if(k == -1 || list2.get(i).equals(list2.get(k))) {
                k++;
                i++;
                if(list2.get(i).equals(list2.get(k))) {
                    next[i] = next[k];
                } else {
                    next[i] = k;
                }
            } else {
                k = next[k];
            }
        }
        k = 0;
        i = 0;
        while(i < list1.size() && k < list2.size()) {
            if(k == -1 || list1.get(i).equals(list2.get(k))) {
                i++;
                k++;
            }else {
                k = next[k];
            }
        }
        return k == list2.size();
    }
}
```

时间复杂度：O（s+t）

### 583. 两个字符串的删除操作

给定两个单词 *word1* 和 *word2*，找到使得 *word1* 和 *word2* 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

**示例：**

```
输入: "sea", "eat"
输出: 2
解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
```

 链接：https://leetcode-cn.com/problems/delete-operation-for-two-strings/

#### 解法：动态规划

用一个二维数组`dp[i+1][j+1]`表示为使`word1[0...i]=word2[0...j]`相同所需要的最小步数。

- 如果`word1[i] == word2[j]`，那么保留当前这个字母不需要做任何处理，让之前相等即可，`dp[i+1]][j+1]=dp[i][j]`
- 如果`word1[i] != word2[j]`，可以选择删除`word1[i]`这个字母，使`word1[0...i-1]=word2[0...j]`，或者删除`word2[j]`，使`word1[0...i]=word2[0...j-1]`，`dp[i+1][j+1] = Math.min(dp[i][j+1], dp[i+1][j]) + 1`

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length(), len2 = word2.length();
        char[] W1 = word1.toCharArray();
        char[] W2 = word2.toCharArray();
        int[][] dp = new int[len1+1][len2+1];
        for(int i = 0; i <= len1; i++) {
            dp[i][0] = i;
        }
        for(int i = 0; i<= len2; i++) {
            dp[0][i] = i;
        }
        for(int i = 0; i < len1; i++) {
            for(int j = 0; j < len2; j++) {
                if(W1[i] == W2[j]) {
                    dp[i+1][j+1] = dp[i][j];
                }else {
                    dp[i+1][j+1] = Math.min(dp[i][j+1], dp[i+1][j]) + 1;
                }
            }
        }
        return dp[len1][len2];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（m*n）

空间上还可以优化一下，只用一维数组。

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length(), len2 = word2.length();
        char[] W1 = word1.toCharArray();
        char[] W2 = word2.toCharArray();
        int[] dp = new int[len2+1];
        for(int i = 0; i <= len2; i++) {
            dp[i] = i;
        }
        for(int i = 0; i < len1; i++) {
            int prev = dp[0];
            dp[0] = i+1;
            for(int j = 0; j < len2; j++) {
                int tmp = dp[j+1];
                if(W1[i] == W2[j]) {
                    dp[j+1] = prev;
                }else {
                    dp[j+1] = Math.min(dp[j+1], dp[j]) + 1;
                }
                prev = tmp;
            }
        }
        return dp[len2];
    }
}
```

时间复杂度：O（m*n）

空间复杂度：O（n）

### 589. N叉树的前序遍历

给定一个 N 叉树，返回其节点值的 **前序遍历** 。N 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 `null` 分隔（请参见示例）。

**进阶：**

递归法很简单，你可以使用迭代法完成此题吗?

**示例 1：**

![](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

```
输入：root = [1,null,3,2,4,null,5,6]
输出：[1,3,5,6,2,4]
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

```
输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
输出：[1,2,3,6,7,11,14,4,8,12,5,9,13,10]
```

链接：https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal

#### 解法一：递归

```java
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> preorder(Node root) {        
        if(root == null) return res;
        res.add(root.val);
        for(Node node: root.children) {
            preorder(node);
        }
        return res;
    }
}
```

#### 解法二：迭代

想清楚遍历的顺序。利用双头队列，其实直接用栈也可以，注意的点是加入孩子的时候要逆序。

```java
class Solution {
    public List<Integer> preorder(Node root) {
        List<Integer> res = new ArrayList<>();
        if(root == null) return res;
        Deque<Node> dq = new LinkedList<>();
        dq.push(root);
        while(!dq.isEmpty()) {
            root = dq.pollFirst();
            res.add(root.val);
            if(root.children != null) {
                int size = root.children.size();
                for(int i = size - 1; i >= 0; i--) {
                    dq.addFirst(root.children.get(i));
                }
            }
        }
        return res;
    }
}
```

### 590. N叉树的后序遍历

给定一个 N 叉树，返回其节点值的 **后序遍历** 。N 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 `null` 分隔（请参见示例）。

**进阶：**

递归法很简单，你可以使用迭代法完成此题吗?

**示例 1：**

![](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

```
输入：root = [1,null,3,2,4,null,5,6]
输出：[5,6,3,2,4,1]
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

```
输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
输出：[2,6,14,11,7,3,12,8,4,13,9,10,5,1]
```

链接：https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/

#### 解法一：递归

```java
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> postorder(Node root) {
        if(root == null) return res;
        for(Node node : root.children) {
            postorder(node);
        }
        res.add(root.val);
        return res;
    }
}
```

#### 解法二：迭代

遍历所走的顺序实际上和递归的不同。如果不用`res.add(0,root.val)`而是用`res.add(root.val)`也可以，最后直接翻转数组即可。

```java
class Solution {
    public List<Integer> postorder(Node root) {
        List<Integer> res = new ArrayList<>();
        if(root == null) return res;
        Stack<Node> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()) {
            root = stack.pop();
            res.add(0, root.val);
            for(Node node : root.children) {
                stack.push(node);
            }
        }
        return res;
    }
}
```

### 594. 最长和谐子序列

和谐数组是指一个数组里元素的最大值和最小值之间的差别 **正好是 `1`** 。现在，给你一个整数数组 `nums` ，请你在所有可能的子序列中找到最长的和谐子序列的长度。数组的子序列是一个由数组派生出来的序列，它可以通过删除一些元素或不删除元素、且不改变其余元素的顺序而得到。

**示例 1：**

```
输入：nums = [1,3,2,2,5,2,3,7]
输出：5
解释：最长的和谐子序列是 [3,2,2,2,3]
```

**示例 2：**

```
输入：nums = [1,2,3,4]
输出：2
```

**示例 3：**

```
输入：nums = [1,1,1,1]
输出：0
```

链接：https://leetcode-cn.com/problems/longest-harmonious-subsequence

#### 解法一：排序 滑动窗口

先用库函数对数组进行排序，然后就是找差为1的最长**连续**子数组长度。有了连续这个条件，就可以利用到滑动窗口了。

```java
class Solution {
    public int findLHS(int[] nums) {
        if(nums.length == 1) return 0;
        Arrays.sort(nums);
        int begin = 0;//保存数组的索引，而不是保存值，会省去很多麻烦
        int res = 0;
        for(int i = 0; i < nums.length; i++) {
            while(nums[i] - nums[begin] > 1) {
                begin++;
            }
            if(nums[i] - nums[begin] == 1) res = Math.max(res, i - begin + 1);
        }
        return res;
    }
}
```

时间复杂度：O（nlogn）

#### 解法二：哈希表

先扫描一遍数组，把每个数值与其对应的出现频次存储到哈希表中。

再遍历哈希表中的每一对键值对，找比它大1的数值出现频次。

```java
class Solution {
    public int findLHS(int[] nums) {
        if(nums.length == 1) return 0;
        HashMap<Integer, Integer> hm = new HashMap<>();
        for(int i = 0; i < nums.length; i++) {
            hm.put(nums[i],hm.getOrDefault(nums[i],0) + 1);
        }
        int res = 0;
        for(int num : hm.keySet()) {
            res = Math.max(res, hm.get(num) + hm.getOrDefault(num+1, 0));
        }
        return res;
    }
}
```

时间复杂度：O（n）

### 605. 种花问题

假设有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花不能种植在相邻的地块上，它们会争夺水源，两者都会死去。

给你一个整数数组 `flowerbed` 表示花坛，由若干 `0` 和 `1` 组成，其中 `0` 表示没种植花`1` 表示种植了花。另有一个数 `n` ，能否在不打破种植规则的情况下种入 `n` 朵花？能则返回 `true` ，不能则返回 `false`。

**示例 1：**

```
输入：flowerbed = [1,0,0,0,1], n = 1
输出：true
```

**示例 2：**

```
输入：flowerbed = [1,0,0,0,1], n = 2
输出：false
```

链接：https://leetcode-cn.com/problems/can-place-flowers

#### 解法：贪心

从前往后遍历，一旦找到一个两边都没有种花的地就立刻种上。

```java
class Solution {
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int len = flowerbed.length;
        if(n > len/2+1) return false;
        int sum = 0;
        for(int i = 0; i < len; i++) {
            if(flowerbed[i] == 1) continue;
            //先处理在数组两端的情况，只需要一边有空地即可
            if(i == 0) {
                if(len == 1 || (1 < len && flowerbed[1] == 0)) {
                    sum++;
                    flowerbed[i] = 1;
                }
            }else if(i == len-1) {
                if(i-1 >= 0 && flowerbed[i-1] == 0) {
                    sum++;
                    flowerbed[i] = 1;
                }
            }else if(flowerbed[i-1] == 0 && flowerbed[i+1] == 0) {
                //如果两边都有地
                sum++;
                flowerbed[i] = 1;
            }
        }
        return sum >= n;
    }
}
```

### 617. 合并二叉树

给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则**不为** NULL 的节点将直接作为新二叉树的节点。

**示例 1:**

```
输入: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
```

**注意:** 合并必须从两个树的根节点开始。
链接：https://leetcode-cn.com/problems/merge-two-binary-trees

#### 解法一：深度优先搜索

对于两个都是非空的节点，直接让两个值相加就可以，如果遇到某一个节点，一颗树上的是空，另一颗树不是，其实可以直接返回非空的那个节点（如果题目不允许修改树结构，必须生成一个新的树，是不可以这样做的）。

下面的写法是直接覆盖在`Tree1`上的。

```java
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if(root1 == null){
            return root2;
        }
        if(root2 == null){
            return root1;
        }
        root1.val = root1.val + root2.val;
        root1.left = mergeTrees(root1.left, root2.left);
        root1.right = mergeTrees(root1.right, root2.right);
        return root1;
    }
}
```

#### 解法二：广度优先搜索

BFS的写法就要繁琐很多了。但是代码还是好理解的。

```java
class Solution {
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if (t1 == null) {
            return t2;
        }
        if (t2 == null) {
            return t1;
        }
        TreeNode merged = new TreeNode(t1.val + t2.val);
        //用三个队列分别保存三棵树的遍历节点
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        Queue<TreeNode> queue1 = new LinkedList<TreeNode>();
        Queue<TreeNode> queue2 = new LinkedList<TreeNode>();
        queue.offer(merged);
        queue1.offer(t1);
        queue2.offer(t2);
        while (!queue1.isEmpty() && !queue2.isEmpty()) {
            TreeNode node = queue.poll(), node1 = queue1.poll(), node2 = queue2.poll();
            TreeNode left1 = node1.left, left2 = node2.left, right1 = node1.right, right2 = node2.right;
            if (left1 != null || left2 != null) {
                if (left1 != null && left2 != null) {
                    TreeNode left = new TreeNode(left1.val + left2.val);
                    node.left = left;
                    queue.offer(left);
                    queue1.offer(left1);
                    queue2.offer(left2);
                } else if (left1 != null) {
                    node.left = left1;
                } else if (left2 != null) {
                    node.left = left2;
                }
            }
            if (right1 != null || right2 != null) {
                if (right1 != null && right2 != null) {
                    TreeNode right = new TreeNode(right1.val + right2.val);
                    node.right = right;
                    queue.offer(right);
                    queue1.offer(right1);
                    queue2.offer(right2);
                } else if (right1 != null) {
                    node.right = right1;
                } else {
                    node.right = right2;
                }
            }
        }
        return merged;
    }
}
```

### 633. 平方数之和

给定一个非负整数 `c` ，你要判断是否存在两个整数 `a` 和 `b`，使得 `a^2 + b^2 = c` 。

**示例 1：**

```
输入：c = 5
输出：true
解释：1 * 1 + 2 * 2 = 5
```

**示例 2：**

```
输入：c = 3
输出：false
```

**示例 3：**

```
输入：c = 4
输出：true
```

**示例 4：**

```
输入：c = 2
输出：true
```

**示例 5：**

```
输入：c = 1
输出：true
```

链接：https://leetcode-cn.com/problems/sum-of-square-numbers

#### 解法一：双指针

一个指针指向0，另一个指向$\sqrt{c}$，如果两个数的平方和大于$c$，就左移右边的指针，如果小，就右移左边的指针。

为了防止溢出，这里用long。

```java
class Solution {
    public boolean judgeSquareSum(int c) {
        long left = 0;
        long right = (long) Math.sqrt(c);
        while (left <= right) {
            long sum = left * left + right * right;
            if (sum == c) {
                return true;
            } else if (sum > c) {
                right--;
            } else {
                left++;
            }
        }
        return false;
    }
}
```

时间复杂度：O（$\sqrt{c}$）

#### 解法二：费马和平方定理

> 一个非负整数 c 如果能够表示为两个整数的平方和，当且仅当 c 的所有形如 4k + 3的**质因子**的幂均为偶数。

因此我们需要对 c 进行**质因数分解**，再判断**所有**形如 4k + 3 的质因子的幂是否均为偶数即可。

```java
class Solution {
    public boolean judgeSquareSum(int c) {
        for (int base = 2; base * base <= c; base++) {
            // 如果不是因子，枚举下一个
            if (c % base != 0) {
                continue;
            }
            // 计算 base 的幂
            int exp = 0;
            while (c % base == 0) {
                c /= base;
                exp++;
            }
            // 根据 Sum of two squares theorem 验证
            if (base % 4 == 3 && exp % 2 != 0) {
                return false;
            }
        }
      	// 例如 11 这样的用例，由于上面的 for 循环里 base * base <= c ，base == 11 的时候不会进入循环体
      	// 因此在退出循环以后需要再做一次判断
        return c % 4 != 3;
    }
}
```

#### 解法三：规律

根据公式$1+3+5+...+(2n+1) = (1+2n+1)*n/2 = n^2$，可以知道，一个平方数肯定是可以拆解成一堆奇数的，所以两个平方数的和，一定也是可以拆解成一堆奇数。

```java
class Solution {
    public boolean judgeSquareSum(int c) {
        int a = (int)Math.sqrt(c);
        if(c == a*a) return true;
        for(int i = 1; i <= c; i += 2) {
            c -= i;
            int j = (int)Math.sqrt(c);
            if(j*j == c) return true;
        }
        return false;
    }
}
```

### 637. 二叉树的层平均值

给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。

**示例 1：**

```
输入：
    3
   / \
  9  20
    /  \
   15   7
输出：[3, 14.5, 11]
解释：
第 0 层的平均值是 3 ,  第1层是 14.5 , 第2层是 11 。因此返回 [3, 14.5, 11] 。
```

链接：https://leetcode-cn.com/problems/average-of-levels-in-binary-tree

#### 解法：广度优先搜索

按照层序遍历来做。

```java
class Solution {
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> result = new ArrayList<>();
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        while(!que.isEmpty()) {
            int size = que.size();
            double sum = 0;
            for(int i = 0; i < size; i++) {
                TreeNode node = que.poll();
                sum += node.val;
                TreeNode left = node.left, right = node.right;
                if(left != null) que.offer(left);
                if(right != null) que.offer(right);
            }
            result.add(sum/size);
        }
        return result;
    }
}
```

也可以按照深度优先搜索来做，但是有点麻烦，需要维护两个数组，一个记录每一层的节点值总和，一个记录层节点数。最后再计算得到结果数组。

### 647. 回文子串

给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

**示例 1：**

```
输入："abc"
输出：3
解释：三个回文子串: "a", "b", "c"
```

**示例 2：**

```
输入："aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
```

链接：https://leetcode-cn.com/problems/palindromic-substrings

#### 解法一：暴力法

通过切割得到不同的子串，分别判断是否为回文子串

```java
class Solution {
    public int countSubstrings(String s) {
        int len = s.length();
        if(len <= 1) return len;
        int count = 0;
        for(int i = 0; i < len; i++) {
            for(int j = i; j < len; j++) {
                count += isPalindrome(s, i, j);
            }
        }
        return count;
    }
    private int isPalindrome(String s, int start, int end) {
        if(start == end) return 1;
        int left = start, right = end;
        while(left < right) {
            if(s.charAt(left++) != s.charAt(right--)) return 0;
        }
        return 1;
    }
}
```

时间复杂度：O（n^2^）

这样必然会有重复的计算，例如在找`abcba`是否为回文串的时候，就已经查询过`bcb`是否为回文串了。回文串的数量查询如果不是从两头开始而是从中间扩散，可以避免重复计算。

#### 解法二：中心扩散法

以一个字符或两个字符为中心，向两边扩散来找有多少个回文子串。

```java
class Solution {
    public int countSubstrings(String s) {
        int len = s.length();
        if(len <= 1) return len;
        int count = 0;
        for(int i = 0; i < len; i++) {
            count += isPalindrome(s, i, i); //奇数个字符 
            count += isPalindrome(s, i, i + 1); //偶数个字符
        }
        return count;
    }
    private int isPalindrome(String s, int start, int end) {
        if(start < 0 || start >= s.length() || end < 0 || end >= s.length()) return 0; //不在合法范围内
        int left = start, right = end, count = 0;
        while(left >= 0 && right < s.length()) {
            if(s.charAt(left--) != s.charAt(right++)) return count;//由中间向两边延伸
            count++;
        }
        return count;
    }
}
```

时间复杂度：O（n^2^）

虽然时间复杂度没变，但是已经避免了重复计算。

#### 解法三：动态规划

用一个布尔数组`dp[i][j]`记录从`i`到`j`能否构成一个回文串。

遍历的顺序是先按照长度为1的子串，在按照长度为2，再按照长度为3...如此递增。

如果`dp[i+1][j-1]=true`且`s[i]=s[j]`，那么`dp[i][j]=true`。

```java
class Solution {
    public int countSubstrings(String s) {
        int len = s.length();
        if(len <= 1) return len;
        int count = 0;
        boolean[][] dp = new boolean[len][len];
        for(int l = 1; l <= len; l++) {
            for(int i = 0; i < len-l+1; i++) {
                if(l == 1) {
                    dp[i][i] = true;
                    count++;
                }else {
                    if(s.charAt(i) == s.charAt(i+l-1) && (l == 2 || dp[i+1][i+l-2])) {
                        count++;
                        dp[i][i+l-1] = true;
                    }
                }
            }
        }
        return count;
    }
}
```

时间复杂度：O（n^2^）

#### 解法四：Manachar算法

Manacher 算法是在线性时间内求解最长回文子串的算法。[具体看](https://leetcode-cn.com/problems/palindromic-substrings/solution/hui-wen-zi-chuan-by-leetcode-solution/)

面对奇数长度和偶数长度的问题，它的处理方式是在所有的相邻字符中间插入 #，例如`abc`会被处理成`#a#b#c#`。这样可以保证所有找到的回文串都是奇数长度的，以任意一个字符为回文中心，既可以包含原来的奇数长度的情况，也可以包含原来偶数长度的情况。在开头加上`$`结尾加上`!`这样当一个字符串本身是回文串的时候，遇到这两个字符就会停止，下标一定不会越界。

用一个参数`f[i]`记录以字符`i`为中心扩散所能达到的最长子串半径。用`iMax`记录能到达最远端的回文子串中心，`rMax`记录右端端点。

```java
class Solution {
    public int countSubstrings(String s) {
        int len = s.length();
        if(len <= 1) return len;
        //---------------处理字符串----------------
        StringBuilder sb = new StringBuilder();
        sb.append("$#");
        for(int i = 0; i < len; i++) {
            sb.append(s.charAt(i));
            sb.append("#");
        }
        sb.append("!");
        s = sb.toString();
        len = s.length();
        //----------------中心扩散计算回文子串的长度------------------
        int count = 0;
        int[] f = new int[len];
        int iMax = 0, rMax = 0;
        for(int i = 1; i < len-1; i++) {
            //初始化f[i]，如果i在rMax的范围内，可以利用i相对于iMax的对称位置值
            //如果不在该范围内，初始化为1
            f[i] = i <= rMax ? Math.min(f[iMax*2 - i], rMax - i + 1) : 1;
            while(s.charAt(i-f[i]) == s.charAt(i+f[i])) {
                f[i]++;
            }
            if(i+f[i]-1 > rMax) {
                rMax = i+f[i]-1;
                iMax = i;                
            }
            //f[i]-1其实就是回文子串真正的长度（去掉后来添加的符号）
            //回文子串的个数也是由这个最长长度的一半决定的
            //对于f[i]-1是奇数的情况，它除以2后还需要+1取整，所以可以直接用f[i]/2
            count += f[i]/2;
        }
        return count;
    }
}
```

时间复杂度：O（n）

### 650. 只有两个键的键盘

最初在一个记事本上只有一个字符 'A'。你每次可以对这个记事本进行两种操作：

1. `Copy All` (复制全部) : 你可以复制这个记事本中的所有字符(部分的复制是不允许的)。
2. `Paste` (粘贴) : 你可以粘贴你**上一次**复制的字符。

给定一个数字 `n` 。你需要使用最少的操作次数，在记事本中打印出**恰好** `n` 个 'A'。输出能够打印出 `n` 个 'A' 的最少操作次数。

**示例 1:**

```
输入: 3
输出: 3
解释:
最初, 我们只有一个字符 'A'。
第 1 步, 我们使用 Copy All 操作。
第 2 步, 我们使用 Paste 操作来获得 'AA'。
第 3 步, 我们使用 Paste 操作来获得 'AAA'。
```

**说明:**

n 的取值范围是 [1, 1000] 。

链接：https://leetcode-cn.com/problems/2-keys-keyboard

#### 解法一：数学

将所有操作分成以 `copy` 为首的多组，形如 `(copy, paste, ..., paste)`，再使用 `C` 代表 `copy`，`P` 代表 `paste`。例如操作 `CPPCPPPPCP` 可以分为 `[CPP][CPPPP][CP]` 三组。每一组的操作步数为g1，g2......，可以得到最后的字符串长度为$g1*g2*g3...$

题目要求找出$n=g1*g2*...$的操作步数，就是要求n可以分解成哪几个数相乘。

```java
class Solution {
    public int minSteps(int n) {
        int ans = 0, d = 2;
        while (n > 1) {
            while (n % d == 0) {
                ans += d;
                n /= d;
            }
            d++;
        }
        return ans;
    }
}
```

时间复杂度：最好情况是O（$\sqrt{N}$），最坏情况是O（N）

#### 解法二：分治 递归

通过思考可以发现，就是看一个数可以拆解成哪几个数的乘积，如果是质数就返回其本身。

如果不是质数，假设可以得到 a*b=n ，取出其中较小的数，作为此次的操作数，继续去找较大数的分解。

```java
class Solution {
    public int minSteps(int n) {
        if(n == 1) return 0;
        if(n <= 5) return n;
        
        if(n % 2 == 0) {
            return 2 + minSteps(n/2);
        }else {
            int d = n-1;
            while(n % d != 0) {
                d--;
            }
            if(d == 1) return n;
            return n/d + minSteps(d);
        }
    }
}
```

#### 解法三：动态规划

这题还可以这么理解，如果要得到3个`A`，是对一个`A`进行dp[3]次操作。

如果是要得到6个`A`，可以将它分成3组2个`A`，也就是先得到一组`AA`，再将它看作像一个`A`那样做dp[3]次操作即可得到6个。所以`dp[i]=dp[i/j]+dp[j]`。

```java
class Solution {
    public int minSteps(int n) {
        if(n == 1) return 0;
        if(n <= 5) return n;
        int[] dp = new int[n+1];
        for(int i = 2; i <= n; i++) {
            dp[i] = i;
            for(int j = 2; j*j <= n; j++) {
                if(i%j == 0) {
                    //所有素数分解的方式都会得到同样的结果
                    dp[i] = dp[i/j] + dp[j];
                    break;
                }
            }
        }
        return dp[n];
    }
}
```

### 654. 最大二叉树

给定一个不含重复元素的整数数组 `nums` 。一个以此数组直接递归构建的 **最大二叉树** 定义如下：

1. 二叉树的根是数组 `nums` 中的最大元素。
2. 左子树是通过数组中 **最大值左边部分** 递归构造出的最大二叉树。
3. 右子树是通过数组中 **最大值右边部分** 递归构造出的最大二叉树。

返回有给定数组 `nums` 构建的 **最大二叉树** 。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/12/24/tree1.jpg)

```
输入：nums = [3,2,1,6,0,5]
输出：[6,3,5,null,2,0,null,null,1]
解释：递归调用如下所示：
- [3,2,1,6,0,5] 中的最大值是 6 ，左边部分是 [3,2,1] ，右边部分是 [0,5] 。
    - [3,2,1] 中的最大值是 3 ，左边部分是 [] ，右边部分是 [2,1] 。
        - 空数组，无子节点。
        - [2,1] 中的最大值是 2 ，左边部分是 [] ，右边部分是 [1] 。
            - 空数组，无子节点。
            - 只有一个元素，所以子节点是一个值为 1 的节点。
    - [0,5] 中的最大值是 5 ，左边部分是 [0] ，右边部分是 [] 。
        - 只有一个元素，所以子节点是一个值为 0 的节点。
        - 空数组，无子节点。
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2020/12/24/tree2.jpg)

```
输入：nums = [3,2,1]
输出：[3,null,2,null,1]
```

链接：https://leetcode-cn.com/problems/maximum-binary-tree

#### 解法一：递归

这题基本上就是翻译题意了，每一次循环都要找到数组中的最大值，创建根节点，然后再分隔成左右两个数组，继续构建树。

```java
class Solution {
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return constructMaximumBinaryTree(nums,0,nums.length-1);
    }
    public TreeNode constructMaximumBinaryTree(int[] nums, int left, int right) {
        if(left > right) return null;
        int[] n = findMax(nums, left, right);
        TreeNode node = new TreeNode(n[0]);
        node.left = constructMaximumBinaryTree(nums,left,n[1]-1);
        node.right = constructMaximumBinaryTree(nums,n[1]+1,right);
        return node;
    }
    private int[] findMax(int[] nums, int left, int right) {
        if(left == right) return new int[]{nums[left],left};
        int max = nums[left];
        int index = left;
        for(int i = left + 1; i <= right; i++) {
            if(nums[i] > max) {
                max = nums[i];
                index = i;
            }
        }
        return new int[]{max,index};
    }
}
```

时间复杂度：调用构造根节点的函数需要调用n次，一般情况的遍历复杂度是O（logn），最坏情况是有序数组，遍历复杂度O（n），所以最坏情况下的时间复杂度是 O（n^2^），最好情况是O（nlogn）。

#### 解法二：回溯

看到了一种[回溯的写法](https://leetcode-cn.com/problems/maximum-binary-tree/solution/654zui-da-er-cha-shu-by-fkczq17mot-vxq6/)，省去了每次都要遍历当前数组寻找最大值的情况。

从前往后遍历数组，如果当前值比前一个值要小，就直接加入右节点。如果大于前一个值，那么就要弹栈，一直到遇到前一个值不小于它的情况，将当前值加入这个前节点的右边，将原右边的树接到当前节点的左边。

```java
class Solution {
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        //的值nums数组最大值不超过1000，构建一个伪头节点，值为最大，保证真正的最大树在该节点的右树上。
        TreeNode fakeRoot = new TreeNode(1001);
        constructMaximumBinaryTree(nums, fakeRoot, 0);
        return fakeRoot.right;
    }
    public int constructMaximumBinaryTree(int[] nums, TreeNode prev, int index) {
        while(index >= 0 && index < nums.length) {
            if(nums[index] < prev.val) {
                TreeNode cur = new TreeNode(nums[index]);
                TreeNode tmp = prev.right;
                prev.right = cur;
                cur.left = tmp;
                index = constructMaximumBinaryTree(nums, cur, index+1);
            }else {
                return index;
            }
        }
        return -1;
    }
}
```

时间复杂度：最好情况是该数组是一个递减数组，不需要往前回溯寻找最大值，时间复杂度为O（n），最坏情况是为一个递增数组，时间复杂度为O（n^2^）。

### 669. 修剪二叉搜索树

给你二叉搜索树的根节点 `root` ，同时给定最小边界`low` 和最大边界 `high`。通过修剪二叉搜索树，使得所有节点的值在`[low, high]`中。修剪树不应该改变保留在树中的元素的相对结构（即，如果没有被移除，原有的父代子代关系都应当保留）。可以证明，存在唯一的答案。

所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/09/09/trim1.jpg)

```
输入：root = [1,0,2], low = 1, high = 2
输出：[1,null,2]
```

**示例 2：**

![](https://assets.leetcode.com/uploads/2020/09/09/trim2.jpg)

```
输入：root = [3,0,4,null,2,null,null,1], low = 1, high = 3
输出：[3,2,null,1]
```

**示例 3：**

```
输入：root = [1], low = 1, high = 2
输出：[1]
```

**示例 4：**

```
输入：root = [1,null,2], low = 1, high = 3
输出：[1,null,2]
```

**示例 5：**

```
输入：root = [1,null,2], low = 2, high = 4
输出：[2]
```

链接：https://leetcode-cn.com/problems/trim-a-binary-search-tree

#### 解法：递归

```java
class Solution {
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if(root == null) return null;
        if(root.val < low) { 
            //如果当前值小于最小边界值，那么就用此节点的右孩子顶替此节点
            return trimBST(root.right,low,high);
        }else if(root.val > high) {
            //如果当前值大于最大边界值，那么就用此节点的左孩子顶替此节点
            return trimBST(root.left,low,high);
        }else { //如果是在给定的边界内，那么就对他们的左右孩子进行递归修剪
            root.left = trimBST(root.left,low,high);
            root.right = trimBST(root.right,low,high);   
            return root;     
        }
    }
}
```

时间复杂度：O（n）

### 674. 最长连续递增序列

给定一个未经排序的整数数组，找到最长且 **连续递增的子序列**，并返回该序列的长度。**连续递增的子序列** 可以由两个下标 `l` 和 `r`（`l < r`）确定，如果对于每个 `l <= i < r`，都有 `nums[i] < nums[i + 1]` ，那么子序列 `[nums[l], nums[l + 1], ..., nums[r - 1], nums[r]]` 就是连续递增子序列。

**示例 1：**

```
输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。
```

 **示例 2：**

```
输入：nums = [2,2,2,2,2]
输出：1
解释：最长连续递增序列是 [2], 长度为1。
```

链接：https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence

#### 解法：贪心

一道简单题，毫无技术可言了...

```java
class Solution {
    public int findLengthOfLCIS(int[] nums) {
        int len = nums.length;
        if(len <= 1) return len;
        int max = 1, now = 1;
        for(int i =1; i < len; i++) {
            if(nums[i] > nums[i-1]) {
                now++;
            }else {
                max = Math.max(max, now);
                now = 1;
            }
        }
        return Math.max(max, now);
    }
}
```

### 680. 验证回文字符串 II

给定一个非空字符串 `s`，**最多**删除一个字符。判断是否能成为回文字符串。

**示例 1:**

```
输入: "aba"
输出: True
```

**示例 2:**

```
输入: "abca"
输出: True
解释: 你可以删除c字符。
```

链接：https://leetcode-cn.com/problems/valid-palindrome-ii/

#### 解法：贪心

因为只能删除一个字符，所以在对称位置，如果两个字符不相同的话，只能从这两个字符里删除一个，如果删除这两个字符中的任何一个都不能得到回文子串，那么就一定结果为false。

```java
class Solution {
    public boolean validPalindrome(String s) {
        return valid(s, 0, s.length() - 1, false);
    }

    public boolean valid(String s, int i, int j, boolean isDeep) {
        while (i < j) {
            char a = s.charAt(i), b = s.charAt(j);
            if (a != b ) {
                if(isDeep) return false;
                return valid(s, i, j - 1, true) || valid(s, i + 1, j, true);
            }
            i++;
            j--;
        }
        return true;
    }
}
```

时间复杂度：O（n）

### 695. 岛屿的最大面积

给定一个包含了一些 `0` 和 `1` 的非空二维数组 `grid` 。一个 **岛屿** 是由一些相邻的 `1` (代表土地) 构成的组合，这里的「相邻」要求两个 `1` 必须在水平或者竖直方向上相邻。你可以假设 `grid` 的四个边缘都被 `0`（代表水）包围着。找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 `0` 。)

**示例 1:**

```
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
```

对于上面这个给定矩阵应返回 6。注意答案不应该是 11 ，因为岛屿只能包含水平或垂直的四个方向的 1 。

**示例 2:**

```
[[0,0,0,0,0,0,0,0]]
```

对于上面这个给定的矩阵, 返回 0。
链接：https://leetcode-cn.com/problems/max-area-of-island

#### 解法一：深度优先搜索

找到为土地的方格，然后找到与它连通的网格，一直递归直到没有连通网格再返回。

这里可以通过另一个数组来存储该网格是否被搜索过。

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        if(grid.length == 0) return 0;
        int height = grid.length, width = grid[0].length;
        boolean[][] marked = new boolean[height][width];
        int count = 0;
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                int sum = 0;
                sum = dfs(grid, marked, i, j, sum);
                if(sum > count) count = sum;  
            }
        }
        return count;
    }
    private int dfs(int[][] grid, boolean[][] marked, int i, int j, int sum) {
        if(grid[i][j] == 1) {
            marked[i][j] = true;
            sum++;
            //上
            if(i > 0 && !marked[i - 1][j]) sum = dfs(grid, marked, i - 1, j, sum);
            //下
            if(i < grid.length - 1 && !marked[i + 1][j]) sum = dfs(grid, marked, i + 1, j, sum);
            //左
            if(j > 0 && !marked[i][j - 1]) sum = dfs(grid, marked, i, j - 1, sum);
            //右
            if(j < grid[0].length - 1 && !marked[i][j + 1]) sum = dfs(grid, marked, i, j + 1, sum);
        }
        return sum;
    }
}
```

这里也可以通过直接修改数组值来达到遍历一次后不会再重复搜索该网格的目的。

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        if(grid.length == 0) return 0;
        int height = grid.length, width = grid[0].length;
        int count = 0;
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                int sum = 0;
                sum = dfs(grid, i, j, sum);
                if(sum > count) count = sum;  
            }
        }
        return count;
    }
    private int dfs(int[][] grid, int i, int j, int sum) {
        if(grid[i][j] == 1) {
            grid[i][j] = 0;
            sum++;
            //上
            if(i > 0) sum = dfs(grid, i - 1, j, sum);
            //下
            if(i < grid.length - 1) sum = dfs(grid, i + 1, j, sum);
            //左
            if(j > 0) sum = dfs(grid, i, j - 1, sum);
            //右
            if(j < grid[0].length - 1) sum = dfs(grid, i, j + 1, sum);
        }
        return sum;
    }
}
```

#### 解法二：广度优先搜索

用队列存储周围连通的土地坐标，每一次搜索一个土地都先把它所有的连通土地找出来

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        if(grid.length == 0) return 0;
        int height = grid.length, width = grid[0].length;
        int res = 0;
        Queue<Integer> iPos = new LinkedList<>();
        Queue<Integer> jPos = new LinkedList<>();
        int[] dir = {0, -1, 0, 1, 0};
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                if(grid[i][j] == 1) {
                    iPos.offer(i);
                    jPos.offer(j);
                    grid[i][j] = 0;
                    int sum = 0;
                    while(!iPos.isEmpty()) {
                        int m = iPos.poll();
                        int n = jPos.poll();                        
                        sum++;
                        for(int k = 0; k < 4; k++) {
                            int s = m + dir[k];
                            int t = n + dir[k+1];
                            if(s >= 0 && t >= 0 && s < height && t < width && grid[s][t] == 1) {
                                iPos.offer(s);
                                jPos.offer(t);
                                grid[s][t] = 0;
                            }
                        }
                    }
                    res = Math.max(sum, res);
                }
            }
        }
        return res;
    }
}
```

### 696. 计数二进制子串

给定一个字符串 `s`，计算具有相同数量 0 和 1 的非空（连续）子字符串的数量，并且这些子字符串中的所有 0 和所有 1 都是连续的。重复出现的子串要计算它们出现的次数。

**示例 1 :**

```
输入: "00110011"
输出: 6
解释: 有6个子串具有相同数量的连续1和0：“0011”，“01”，“1100”，“10”，“0011” 和 “01”。

请注意，一些重复出现的子串要计算它们出现的次数。

另外，“00110011”不是有效的子串，因为所有的0（和1）没有组合在一起。
```

**示例 2 :**

```
输入: "10101"
输出: 4
解释: 有4个子串：“10”，“01”，“10”，“01”，它们具有相同数量的连续1和0。
```

链接：https://leetcode-cn.com/problems/count-binary-substrings

#### 解法一：暴力法

从一个位置`i`出发，往后找到与它相同的字符个数，然后再找与它不同的字符个数，若可以满足相同的需求，`res++`，下一次再从位置`i+1`出发。

```java
class Solution {
    public int countBinarySubstrings(String s) {
        if(s.length() == 1) return 0;
        int count = 0;
        char[] array = s.toCharArray(); 
        for(int i = 0; i < array.length - 1; i++) {
            int j = i+1;
            int c1 = 1, c2 = 0;
            while(j < array.length && array[j] == array[i]) {
                c1++;
                j++;
            }
            while(j < array.length && c2 < c1 && array[j] != array[i]) {
                c2++;
                j++;
            }
            if(c1 == c2) count++;
        }
        return count;
    }
}
```

时间复杂度：O（n^2^）



但是这样做会超出时间限制，因为中间做了很多重复计算，比如计算"0011"从位置`0`出发往后搜索得到两个“0”然后又计算得到可以有两个“1”，这时就已经可以计算得出这个字符串只能得到两个满足题目要求的子串。

做了优化。

- 当`n1=n2`时，用`array[i..i+n1-1]`中任意一个字符作开头，必然可以构成`n1`个子串。
- 当`n1>n2`时，说明此时`j==array.length`。用`array[i..i+n1-1]`中任意一个字符作开头，一定可以构成`n2`个子串。

用`array[i+n1]`之后字符作开头是未知的，所以下一次要从`i+n1`位置开始。

```java
class Solution {
    public int countBinarySubstrings(String s) {
        if(s.length() == 1) return 0;
        int count = 0;
        char[] array = s.toCharArray(); 
        for(int i = 0; i < array.length - 1; i++) {
            int j = i+1;
            int n1 = 1, n2 = 0;
            char c = array[i];
            while(j < array.length && array[j] == c) {
                n1++;
                j++;
            }
            while(j < array.length && n2 < n1 && array[j] != c) {
                n2++;
                j++;
            }
            count += n2;
            i = i + n1 - 1;            
        }
        return count;
    }
}
```

时间复杂度：O（n）

上面的代码还可以做修改，不需要用while，直接从前往后遍历，计算与当前字符相同的连续字符个数，以及记录之前遇到的另一个字符的连续个数。

```java
class Solution{
    public int countBinarySubstrings(String s) {
    	int preLen = 0, curLen = 1, count = 0;
    	char[] array = s.toCharArray();
    	for (int i = 1; i < array.length; i++) {
        	if (array[i] == array[i-1]) {
            	curLen++;
        	} else {
            	preLen = curLen;
            	curLen = 1;
       	 	}

        	if (preLen >= curLen) {
            	count++;
        	}
    	}
    	return count;
	}
}
```

时间复杂度：O（n）

#### 解法二：中心扩散

找到前后两个字符不同的位置，以这个为中心向两边扩散计算长度。

```java
class Solution {
    public int countBinarySubstrings(String s) {
        if(s.length() == 1) return 0;
        int count = 0;
        char[] array = s.toCharArray(); 
        for(int i = 0; i < s.length() - 1; i++) {
            if(array[i] != array[i+1]) count += (countBinarySubstrings(array, i, i+1) + 1); 
        }
        return count;
    }
    private int countBinarySubstrings(char[] array, int start, int end) {
        int count = 0;
        char l = array[start];
        char r = array[end];
        while((--start) >= 0 && (++end) < array.length) {
            if(array[start] != l || array[end] != r) return count;
            count++;
        }
        return count;
    }
}
```

### 700. 二叉搜索树中的搜索

给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。

例如，

给定二叉搜索树:

        4
       / \
      2   7
     / \
    1   3
    和值: 2


你应该返回如下子树:

      2     
     / \   
    1   3
在上述示例中，如果要找的值是 5，但因为没有节点值为 5，我们应该返回 NULL。
链接：https://leetcode-cn.com/problems/search-in-a-binary-search-tree



最基本的二叉搜索树用法了，没什么可说的

```java
class Solution {
    public TreeNode searchBST(TreeNode root, int val) {
        if(root == null) return null;
        int nowV = root.val;
        if(nowV == val) return root;
        if(nowV < val) return searchBST(root.right, val);
        return searchBST(root.left, val);
    }
}
```

### 701. 二叉搜索树中的插入操作

给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。返回插入后二叉搜索树的根节点。 输入数据 **保证** ，新值和原始二叉搜索树中的任意节点值都不同。**注意**，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。你可以返回 **任意有效的结果** 。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/10/05/insertbst.jpg)

```
输入：root = [4,2,7,1,3], val = 5
输出：[4,2,7,1,3,5]
```

解释：另一个满足题目要求可以通过的树是：

![](https://assets.leetcode.com/uploads/2020/10/05/bst.jpg)

**示例 2：**

```
输入：root = [40,20,60,10,30,50,70], val = 25
输出：[40,20,60,10,30,50,70,null,null,25]
```

**示例 3：**

```
输入：root = [4,2,7,1,3,null,null,null,null,null,null], val = 5
输出：[4,2,7,1,3,5]
```

链接：https://leetcode-cn.com/problems/insert-into-a-binary-search-tree

#### 解法：递归

当前节点小于val，那么val节点一定在右子树上；若大于，则在左子树上。

直到遇到null节点，这里就是val节点应该在的位置。

```java
class Solution {
    public TreeNode insertIntoBST(TreeNode root, int val) {
        if(root == null) return new TreeNode(val);
        if(root.val < val) {
            root.right = insertIntoBST(root.right, val);
        }else {
            root.left = insertIntoBST(root.left, val);
        }
        return root;
    }
}
```

### 707. 设计链表

设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：`val` 和 `next`。`val` 是当前节点的值，`next` 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 `prev` 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。

在链表类中实现这些功能：

- get(index)：获取链表中第 `index` 个节点的值。如果索引无效，则返回`-1`。
- addAtHead(val)：在链表的第一个元素之前添加一个值为 `val` 的节点。插入后，新节点将成为链表的第一个节点。
- addAtTail(val)：将值为 `val` 的节点追加到链表的最后一个元素。
- addAtIndex(index,val)：在链表中的第 `index` 个节点之前添加值为 `val` 的节点。如果 `index` 等于链表的长度，则该节点将附加到链表的末尾。如果 `index` 大于链表长度，则不会插入节点。如果`index`小于0，则在头部插入节点。
- deleteAtIndex(index)：如果索引 `index` 有效，则删除链表中的第 `index` 个节点。

**示例：**

```
MyLinkedList linkedList = new MyLinkedList();
linkedList.addAtHead(1);
linkedList.addAtTail(3);
linkedList.addAtIndex(1,2);   //链表变为1-> 2-> 3
linkedList.get(1);            //返回2
linkedList.deleteAtIndex(1);  //现在链表是1-> 3
linkedList.get(1);            //返回3
```

链接：https://leetcode-cn.com/problems/design-linked-list

#### 解法一：单链表

规定一个Node结构，包含next和val。

链表中除了有成员变量Head以外，为了方便也记录下Tail，还要记录节点数量。

- get(index)：先要验证index是否合法，如果合法就往后寻找。
- addAtHead(val)：如果链表中没有头，那么要将Head和Tail都指向这个新创建的节点；如果有头节点，那么把新节点的next指向原头节点，然后再将Head重新指向这个节点。
- addAtTail(val)：如果链表暂且是空的，那么要将Head和Tail都指向新创建的节点；如果不为空，就把原Tail的next指向新创建的节点，再把Tail重新指向这个节点。
- addAtIndex(index,val)：如果index不合法直接不插入。如果是要插入头节点，那么可以直接调用addAtHead(val)，插入尾节点也可以直接调用。如果是要在中间插入，就要先找到位置。
- deleteAtIndex(index)：如果有效，在头节点的话要注意头节点指针的变动，尾部要注意尾节点的变动，如果头尾指针是重叠的，也要做特殊处理（其实这里不对尾节点处理也可以，因为有count在做辅助判断）。其余的则要先找到中间节点再删除

```java
public class Node {
    int val;
    Node next;
    Node(int x) { val = x; }
    Node() {}
}
class MyLinkedList {
    Node head;
    Node tail;
    int count;
    /** Initialize your data structure here. */
    public MyLinkedList() {
        head = new Node();
        tail = new Node();
        count = 0;
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    public int get(int index) {
        if(index < 0 || index >= count) return -1;
        Node cur = head;
        while(index-- > 0) {
            cur = cur.next;
        }
        return cur.val;
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    public void addAtHead(int val) {
        Node tmp = new Node(val);
        if(count == 0) {
            head = tmp;
            tail = head;
        }else {            
            tmp.next = head;
            head = tmp;
        }
        count++;
    }
    
    /** Append a node of value val to the last element of the linked list. */
    public void addAtTail(int val) {
        Node tmp = new Node(val);
        if(count == 0) {
            tail = tmp;
            head = tail;
        }else {
            tail.next = tmp;
            tail = tmp;
        }
        count++;
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    public void addAtIndex(int index, int val) {
        if(index > count || index < 0) return;//超出范围不加入
        
        Node tmp = new Node(val);
        if(index == 0) {//在头部加入
            tmp.next = head;
            head = tmp;
            count++;
            return;
        }

        if(index == count) {//在尾部加入
            tail.next = tmp;
            tail = tmp;
            count++;
            return;
        }

        Node cur = head;
        Node prev = null;
        while(index-- > 0) {
            prev = cur;
            cur = cur.next;
        }
        prev.next = tmp;
        tmp.next = cur;
        count++;
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    public void deleteAtIndex(int index) {
        if(index >= 0 && index < count) {
            if(index == 0) {
                if(count == 1) tail = null;//这一行不要也可以
                head = head.next;
            }else {
                Node cur = head;
                Node prev = null;
                for(int i = 0; i < index; i++) {
                    prev = cur;
                    cur = cur.next;
                }
                prev.next = cur.next;
                if(index == count - 1) tail = prev;
            }
            count--;
        }
    }
}
```

#### 解法二：双向链表

规定一个Node结构，包含next, prev和val。

链表中除了有成员变量Head以外，为了方便也记录下Tail，还要记录节点数量。

- get(index)：先要验证index是否合法，为了减少寻找次数，如果index大于总数一半，就要从后往前找。
- addAtHead(val)：如果链表中没有头，那么要将Head和Tail都指向这个新创建的节点；如果有头节点，那么把新节点的next指向原头节点，然后再将Head重新指向这个节点。
- addAtTail(val)：如果链表暂且是空的，那么要将Head和Tail都指向新创建的节点；如果不为空，就把原Tail的next指向新创建的节点，记得要把新节点的prev指向原tail，把Tail重新指向这个节点。
- addAtIndex(index,val)：如果index不合法直接不插入。如果是要插入头节点，那么可以直接调用addAtHead(val)，插入尾节点也可以直接调用。如果是要在中间插入，就要先找到位置。不仅要把该节点的prev和next连接好，也要把该节点的前一个节点next和后一个节点的prev做修改。
- deleteAtIndex(index)：如果有效，在头节点的话要注意头节点指针的变动，尾部要注意尾节点的变动，如果头尾指针是重叠的，也要做特殊处理（其实这里不对尾节点处理也可以，因为有count在做辅助判断）。其余的则要先找到中间节点再删除。删除的时候注意要把前一个节点的next指向下一个节点，下一个节点的prev指向前一个节点。

```java
public class Node {
    int val;
    Node next;
    Node prev;
    Node(int x) { val = x; }
    Node() {}
}
class MyLinkedList {
    Node head;
    Node tail;
    int count;
    /** Initialize your data structure here. */
    public MyLinkedList() {
        head = new Node();
        tail = new Node();
        count = 0;
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    public int get(int index) {
        if(index < 0 || index >= count) return -1;
        Node cur = head;
        if(index < count/2){            
            while(index-- > 0) {
                cur = cur.next;
            }
        }else {
            cur = tail;
            index = count-index-1;
            while(index-- > 0) {
                cur = cur.prev;
            }
        }        
        return cur.val;
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    public void addAtHead(int val) {
        Node tmp = new Node(val);
        if(count == 0) {
            head = tmp;
            tail = head;
        }else {            
            tmp.next = head;
            head.prev = tmp;
            head = tmp;
        }
        count++;
    }
    
    /** Append a node of value val to the last element of the linked list. */
    public void addAtTail(int val) {
        Node tmp = new Node(val);
        if(count == 0) {
            tail = tmp;
            head = tail;
        }else {
            tail.next = tmp;
            tmp.prev = tail;
            tail = tmp;
        }
        count++;
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    public void addAtIndex(int index, int val) {
        if(index > count) return;//超出范围不加入
        
        Node tmp = new Node(val);
        if(index <= 0) {
            addAtHead(val);
            return;
        }

        if(index == count) {//在尾部加入
            addAtTail(val);
            return;
        }
        Node cur = head;
        if(index < count/2){
            while(index-- > 0) {
                cur = cur.next;
            }
        }else {
            cur = tail;
            index = count-index-1;
            while(index-- > 0) {
                cur = cur.prev;
            }
        } 
        tmp.next = cur;
        tmp.prev = cur.prev;
        cur.prev = tmp;
        tmp.prev.next = tmp;
        count++;
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    public void deleteAtIndex(int index) {
        if(index >= 0 && index < count) {
            if(index == 0) {
                if(count == 1) tail = null;//这一行不要也可以
                head = head.next;
                if(head != null) head.prev = null;
            }else if(index == count-1) {
                tail = tail.prev;
                tail.next = null;
            }else {
                Node cur = head;
                if(index < count/2){
                    while(index-- > 0) {
                        cur = cur.next;
                    }
                }else {
                    cur = tail;
                    index = count-index-1;
                    while(index-- > 0) {
                        cur = cur.prev;
                    }
                } 
                cur.prev.next = cur.next;
                cur.next.prev = cur.prev;
            }
            count--;
        }
    }
}
```

### 714. 买卖股票的最佳时机含手续费

给定一个整数数组 `prices`，其中第 `i` 个元素代表了第 `i` 天的股票价格 ；非负整数 `fee` 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

**示例 1:**

```
输入: prices = [1, 3, 2, 8, 4, 9], fee = 2
输出: 8
解释: 能够达到的最大利润:  
在此处买入 prices[0] = 1
在此处卖出 prices[3] = 8
在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9
总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
```

链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee

#### 解法：动态规划

因为这题类似的已经遇到过太多次，直接看前面买卖股票的题目就可以了。唯一的不同是在卖出的时候要减掉手续费。

下面直接给出空间最优化版本的代码。

```java
class Solution {
    public int maxProfit(int[] prices, int fee) {
        int len = prices.length;
        if(len <= 1) return 0;
        int chiYou = -prices[0], buChiYou = 0;
        for(int i = 1; i < len; i++) {
            chiYou = Math.max(chiYou, buChiYou - prices[i]);
            buChiYou = Math.max(buChiYou, chiYou + prices[i] - fee);
        }
        return buChiYou;
    }
}
```

### 718. 最长重复子数组

给两个整数数组 `A` 和 `B` ，返回两个数组中公共的、长度最长的子数组的长度。

**示例：**

```
输入：
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
解释：
长度最长的公共子数组是 [3, 2, 1] 。
```

 链接：https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/

#### 解法一：动态规划

规定数组`dp[i+1][j+1]`表示以`nums1[i]` 和`nums2[j]`为结尾的公共子数组长度

- 当`nums1[i] = nums2[j]`时，`dp[i+1][j+1] = dp[i][j]+1`。
- 不相等时，为0

```java
class Solution {
    public int findLength(int[] nums1, int[] nums2) {
        int res = 0;
        int[][] dp = new int[1+nums1.length][1+nums2.length];
        for(int i = 0; i < nums1.length; i++) {
            for(int j = 0; j < nums2.length; j++) {
                if(nums1[i] == nums2[j]) {
                    dp[i+1][j+1] = dp[i][j]+1;
                    res = Math.max(res, dp[i+1][j+1]);
                }
            }
        }
        return res;
    }
}
```

时间复杂度：O（m*n）

空间上还可以优化一下，只用一维数组就可以，但是注意新的结果会造成覆盖，所以遍历顺序要做调整。

```java
class Solution {
    public int findLength(int[] nums1, int[] nums2) {
        int len1 = nums1.length, len2 = nums2.length;
        int[] dp = new int[len2+1];
        int res = 0;
        for(int i = 1; i <= len1; i++) {
            for(int j = len2; j >= 1; j--) {
                //逆序
                if(nums1[i - 1] == nums2[j - 1]) {
                    dp[j] = dp[j-1] + 1;
                    res = Math.max(res, dp[j]);
                }else {
                    //这里一定要置位0
                    dp[j] = 0;
                }
            }
        }
        return res;
    }
}
```

#### 解法二：滑动窗口

算法思路如图：

![](https://pic.leetcode-cn.com/9ed48b9b51214a8bafffcad17356d438b4c969b4999623247278d23f1e43977f-%E9%94%99%E5%BC%80%E6%AF%94%E8%BE%83.gif)

```java
class Solution {
    public int findLength(int[] nums1, int[] nums2) {
        if(nums1.length < nums2.length) return findLength(nums2,nums1);
        int res = 0;
        for(int len = 1; len <= nums2.length; len++) {
            int cur = findLength(nums1,0,len-1,nums2,nums2.length-len,nums2.length-1);
            res = Math.max(cur, res);
        }
        for(int i = 1; i <= nums1.length - nums2.length; i++) {
            int cur = findLength(nums1,i,i+nums2.length-1,nums2,0,nums2.length-1);
            res = Math.max(cur, res);
        }
        for(int len = nums2.length-1; len > 0; len--) {
            int cur = findLength(nums1,nums1.length-len,nums1.length-1,nums2,0,len-1);
            res = Math.max(cur, res);
        }
        return res;
    }
    private int findLength(int[] nums1, int s1, int e1, int[] nums2, int s2, int e2) {
        int res = 0, count = 0;
        int i = s1, j = s2;
        while(i <= e1 && j <= e2) {
            if(nums1[i] == nums2[j]) {
                count++;
                res = Math.max(count, res);
            }else {
                count = 0;
            }
            i++;
            j++;
        }
        return res;
    }
}
```

时间复杂度：O（m*n）

### 738. 单调递增的数字

给定一个非负整数 `N`，找出小于或等于 `N` 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。

（当且仅当每个相邻位数上的数字 `x` 和 `y` 满足 `x <= y` 时，我们称这个整数是单调递增的。）

**示例 1:**

```
输入: N = 10
输出: 9
```

**示例 2:**

```
输入: N = 1234
输出: 1234
```

**示例 3:**

```
输入: N = 332
输出: 299
```

链接：https://leetcode-cn.com/problems/monotone-increasing-digits/

####  解法：贪心

从低往高遍历，如果发现不遵循题目的递增要求，那么就要把高位减1，后面的低位全部换为'9'。为了方便用到了String。

```java
class Solution {
    public int monotoneIncreasingDigits(int N) {
        /**
        * 思路：
        *  从右向左扫描数字，若发现当前数字比其左边一位（较高位）小，
        *  则把其左边一位数字减1，并将该位及其右边的所有位改成9
        */
        String s = String.valueOf(N);
        int len = s.length();
        char[] chars = s.toCharArray();
        int flag = len;
        for(int i = len - 1; i >= 1; i--) {
            if(chars[i] < chars[i-1]) {
                flag = i;
                chars[i-1]--;
            }
        }
        for(int i = flag; i < len; i++) {
            chars[i] = '9';
        }
        return Integer.parseInt(new String(chars));
    }
}
```

### 746. 使用最小花费爬楼梯

数组的每个下标作为一个阶梯，第 `i` 个阶梯对应着一个非负数的体力花费值 `cost[i]`（下标从 `0` 开始）。每当你爬上一个阶梯你都要花费对应的体力值，一旦支付了相应的体力值，你就可以选择向上爬一个阶梯或者爬两个阶梯。请你找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为 0 或 1 的元素作为初始阶梯。

**示例 1：**

```
输入：cost = [10, 15, 20]
输出：15
解释：最低花费是从 cost[1] 开始，然后走两步即可到阶梯顶，一共花费 15 。
```

 **示例 2：**

```
输入：cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
输出：6
解释：最低花费方式是从 cost[0] 开始，逐个经过那些 1 ，跳过 cost[3] ，一共花费 6 。
```

链接：https://leetcode-cn.com/problems/min-cost-climbing-stairs

#### 解法：动态规划

最简单的直接的动态规划

```java
class Solution {
    public int minCostClimbingStairs(int[] cost) {
        int len = cost.length;
        int[] dp = new int[len];
        dp[0] = cost[0];
        dp[1] = cost[1];
        for(int i = 2; i < len; i++) {
            dp[i] = cost[i] + Math.min(dp[i-1], dp[i-2]);
        }
        return Math.min(dp[len-1], dp[len-2]);
    }
}
```

 ### 763. 划分字母区间

字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

**示例：**

```
输入：S = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca", "defegde", "hijhklij"。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
```

链接：https://leetcode-cn.com/problems/partition-labels

#### 解法一：

第一种想法是记录下每个字符第一次出现的位置。

- 如果当前字符从来没出现过，那么直接把该坐标添加入一个记录分割点的数组中，并且用哈希表记录下这个位置。
- 如果字符出现过，那么从后往前查找分割点数组，如果遇到数值在该字符第一次出现位置之后，就要删除。

然后根据分割点数组计算出每一段的字符串长度。

这里为了方便，就直接将分割点数组和最后返回的数组用同一个了。

```java
class Solution {
    public List<Integer> partitionLabels(String S) {
        ArrayList<Integer> result = new ArrayList<>();
        if (S == null || S.length() <= 0) {
            return result;
        }
        int[] lastIndex = new int[26];
        Arrays.fill(lastIndex, -1);
        int length = S.length();
        for (int i = 0; i < length; i++) {
            int cur = lastIndex[S.charAt(i) - 'a'];
            if(cur == -1) {
                result.add(i);
                lastIndex[S.charAt(i) - 'a'] = i;
            }else {
                int j = result.size() - 1;                
                while(j >= 0) {
                    if(result.get(j) <= cur) break;
                    result.remove(j);
                    j--;
                }
            }
        }
        int size = result.size();
        for(int i = 1; i < size; i++) {
            int newValue = result.get(i) - result.get(i-1);
            result.set(i-1, newValue);
        }
        result.set(size-1, length-result.get(size-1));
        return result;
    }
}
```

时间复杂度：O（n）

#### 解法二：

第二种想法是记录下字符最后一次出现的位置。

第二次扫描的时候，更新切割区间范围。只有当此次搜寻的位置恰好是本次切割区间的最后位置，才结束此次切割，放入list中。

```java
class Solution {
    public List<Integer> partitionLabels(String S) {
        ArrayList<Integer> result = new ArrayList<>();
        if (S == null || S.length() <= 0) {
            return result;
        }
        int[] lastIndex = new int[26];
        Arrays.fill(lastIndex, -1);
        int length = S.length();
        for (int i = 0; i < length; i++) {
            lastIndex[S.charAt(i) - 'a'] = i;
        }
        int startIndex = 0;
        int endIndex = 0;
        for (int i = 0; i < length; i++) {
            endIndex = Math.max(endIndex, lastIndex[S.charAt(i) - 'a']);
            if (i == endIndex) {
                result.add(endIndex - startIndex + 1);
                startIndex = endIndex + 1;
            }
        }
        return result;
    }
}
```

### 784. 字母大小写全排列

给定一个字符串`S`，通过将字符串`S`中的每个字母转变大小写，我们可以获得一个新的字符串。返回所有可能得到的字符串集合。

**示例：**

```
输入：S = "a1b2"
输出：["a1b2", "a1B2", "A1b2", "A1B2"]

输入：S = "3z4"
输出：["3z4", "3Z4"]

输入：S = "12345"
输出：["12345"]
```

链接：https://leetcode-cn.com/problems/letter-case-permutation

#### 解法：回溯法

回溯函数返回的条件是索引超过了字符串长度。

对每一个字符都先加入StringBuilder对象，进行下一层递归调用。弹栈的时候要删除该位置的元素。

如果当前字符是一个字母，回溯递归函数在弹栈后，还要把当前字符的大写或小写形式加入，再进行下一层调用。

```java
class Solution {
    List<String> result = new ArrayList<>();
    int len;
    StringBuilder path = new StringBuilder();
    public List<String> letterCasePermutation(String S) {
        //<=57是数字
        len = S.length();
        if(len == 0) return result;
        char[] chars = S.toCharArray();
        backTracking(chars,0);
        return result;
    }
    private void backTracking(char[] chars, int start){
        if(start == len) {
            result.add(path.toString());
            return;
        }
        char c = chars[start];
        path.append(c);
        backTracking(chars, start+1);
        path.deleteCharAt(path.length()-1);
        if(c > 57) {
            //修改大小写
            if(c >= 97) {
                c = (char)(c-32);
            }else {
                c = (char)(c+32);
            }
            path.append(c);
            backTracking(chars, start+1);
            path.deleteCharAt(path.length()-1);
        }
    }
}
```

### 860. 柠檬水找零

在柠檬水摊上，每一杯柠檬水的售价为 `5` 美元。顾客排队购买你的产品，（按账单 `bills` 支付的顺序）一次购买一杯。每位顾客只买一杯柠檬水，然后向你付 `5` 美元、`10` 美元或 `20` 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 `5` 美元。注意，一开始你手头没有任何零钱。如果你能给每位顾客正确找零，返回 `true` ，否则返回 `false` 。

**示例 1：**

```
输入：[5,5,5,10,20]
输出：true
解释：
前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
由于所有客户都得到了正确的找零，所以我们输出 true。
```

**示例 2：**

```
输入：[5,5,10]
输出：true
```

**示例 3：**

```
输入：[10,10]
输出：false
```

**示例 4：**

```
输入：[5,5,10,10,20]
输出：false
解释：
前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
由于不是每位顾客都得到了正确的找零，所以答案是 false。
```

链接：https://leetcode-cn.com/problems/lemonade-change

#### 解法：贪心

对于需要找零的情况，先用最大钞找零，再用小钞找。

```java
class Solution {
    public boolean lemonadeChange(int[] bills) {
        if(bills.length == 0) return true;
        int numFor5 = 0;
        int numFor10 = 0;
        for(int b : bills) {
            if(b == 5) {
                numFor5++;
            }else if(b == 10) {
                numFor5--;
                if(numFor5 < 0) return false;
                numFor10++;
            }else if(b == 20) {
                //先用10元找
                if(numFor10 > 0) {
                    numFor10--;
                    b -= 10;
                }
                //再用5元找
                numFor5 -= (b/5 - 1);
                if(numFor5 < 0) return false;
            }
        }
        return true;
    }
}
```

### 1005. K次取反后最大化的数组和

给定一个整数数组 A，我们**只能**用以下方法修改该数组：我们选择某个索引 `i` 并将 `A[i]` 替换为 `-A[i]`，然后总共重复这个过程 `K` 次。（我们可以多次选择同一个索引 `i`。）以这种方式修改数组后，返回数组可能的最大和。

**示例 1：**

```
输入：A = [4,2,3], K = 1
输出：5
解释：选择索引 (1,) ，然后 A 变为 [4,-2,3]。
```

**示例 2：**

```
输入：A = [3,-1,0,2], K = 3
输出：6
解释：选择索引 (1, 2, 2) ，然后 A 变为 [3,1,0,2]。
```

**示例 3：**

```
输入：A = [2,-3,-1,5,-4], K = 2
输出：13
解释：选择索引 (1, 4) ，然后 A 变为 [2,3,-1,5,4]。
```

链接：https://leetcode-cn.com/problems/maximize-sum-of-array-after-k-negations

#### 解法一：贪心

可以知道如果K=1，那么选择最小的数进行反转，得到的一定是最大和。

所以每次都选择最小的数反转即可。

```java
class Solution {
    public int largestSumAfterKNegations(int[] A, int K) {
        int res = 0;
        int minPos = 0;
        int min = Integer.MAX_VALUE;
        while(K-- > 0) {
            for(int i = 0; i < A.length; i++) {
                if(A[i] < min) {
                    min = A[i];
                    minPos = i;
                }
            }
            A[minPos] = -A[minPos];
            min = Integer.MAX_VALUE;
        }
        for(int a : A) {
            res += a;
        }
        return res;
    }
}
```

时间复杂度：O（kn）

#### 解法二：贪心

在上面的过程中，其实做了好几次排序的操作。能不能只做一次排序呢？

- 如果有负数，那么一定是先对最小的负数，也就是绝对值最大的负数做修改。
- 如果数组内已经没有负数了，而K次数还有剩，若剩余K是偶数，则不需要做操作，若是偶数，则挑选出一个绝对值最小的数（其实就是最小的正数）做负数反转。

这里需要写一个绝对值排序的函数，用快排写。

```java
class Solution {
    public int largestSumAfterKNegations(int[] A, int K) {
        int n = A.length;
        //(1)按绝对值快排
        quickSort(A, 0, n - 1);
        //(2)按绝对值大到小，将负数转换为正数
        for (int i = 0; i < n && K > 0; i++) {
            if (A[i] < 0) {
                A[i] = -A[i];
                K--;
            }
        }
        //(3)如果所有的负数都被转变为正数后
        //还有剩余k次转换，%2余1，则绝对值最小的数字取相反数
        if (K % 2 == 1) {
            A[n - 1] = -A[n - 1];
        }
        //(4)累加
        int sum = 0;
        for (int num : A) {
            sum += num;
        }
        return sum;
    }

    private void quickSort(int[] nums, int left, int right) {
        if (left < right) {
            int partitionIndex = partition(nums, left, right);
            quickSort(nums, left, partitionIndex - 1);
            quickSort(nums, partitionIndex + 1, right);
        }
    }

    private int partition(int[] nums, int left, int right) {
        swap(nums, left, (left + right) >> 1);

        int pivot = left;
        int index = pivot + 1;
        for (int i = index; i <= right; i++) {
            if (Math.abs(nums[i]) > Math.abs(nums[pivot])) {
                swap(nums, i, index);
                index++;
            }
        }
        swap(nums, pivot, --index);
        return index;
    }

    private void swap(int[] nums, int p1, int p2) {
        if (p1 == p2) return;
        int tmp = nums[p1];
        nums[p1] = nums[p2];
        nums[p2] = tmp;
    }
}
```

### 1035. 不相交的线

在两条独立的水平线上按给定的顺序写下 `nums1` 和 `nums2` 中的整数。现在，可以绘制一些连接两个数字 `nums1[i]` 和 `nums2[j]` 的直线，这些直线需要同时满足满足：

-  `nums1[i] == nums2[j]`
- 且绘制的直线不与任何其他连线（非水平线）相交。

请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。以这种方法绘制线条，并返回可以绘制的最大连线数。

**示例 1：**

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/04/28/142.png" style="zoom:20%;" />

```
输入：nums1 = [1,4,2], nums2 = [1,2,4]
输出：2
解释：可以画出两条不交叉的线，如上图所示。 
但无法画出第三条不相交的直线，因为从 nums1[1]=4 到 nums2[2]=4 的直线将与从 nums1[2]=2 到 nums2[1]=2 的直线相交。
```

**示例 2：**

```
输入：nums1 = [2,5,1,2,5], nums2 = [10,5,2,1,5,2]
输出：3
```

**示例 3：**

```
输入：nums1 = [1,3,7,1,7,5], nums2 = [1,9,2,5,1]
输出：2
```

链接：https://leetcode-cn.com/problems/uncrossed-lines

#### 解法：动态规划

用`dp[i][j]`表示`nums1[0...i]`和`nums2[0...j]`之间最多可以连接多少条。

- 如果`nums1[i] == nums2[j]`，`dp[i][j] = dp[i-1][j-1]+1`
- 如果不相等，就不能相连，取一个最大值`dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1])`

```java
class Solution {
    public int maxUncrossedLines(int[] A, int[] B) {
        int len1 = A.length, len2 = B.length;
        int[][] dp = new int[len1 + 1][len2 + 1];
        for(int i = 1; i <= len1; i++) {
            for(int j = 1; j <= len2; j++) {
                if(A[i-1] == B[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[len1][len2];
    }
}
```

空间上还可以再优化，但是不能只用一个一维数组，因为`dp[i][j]`不仅仅取决于`dp[i-1][j-1]`和`dp[i-1][j]`这两个是上一层循环得到结果，还与`dp[i][j-1]`有关，这不再属于上一层，而是本层。所以还需要一个数来保存上一层的结果。

```java
class Solution {
    public int maxUncrossedLines(int[] A, int[] B) {
        int len1 = A.length, len2 = B.length;
        int[] dp = new int[len2 + 1];
        for(int i = 1; i <= len1; i++) {
            int prev = 0;
            for(int j = 1; j <= len2; j++) {
                int temp = dp[j];
                if(A[i-1] == B[j-1]) {
                    dp[j] = prev + 1;
                }else {
                    dp[j] = Math.max(dp[j], dp[j-1]);
                }
                prev = temp;
            }
        }
        return dp[len2];
    }
}
```

### 1047. 删除字符串中的所有相邻重复项

给出由小写字母组成的字符串 `S`，**重复项删除操作**会选择两个相邻且相同的字母，并删除它们。在 S 上反复执行重复项删除操作，直到无法继续删除。在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

**示例：**

```
输入："abbaca"
输出："ca"
解释：
例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。
```

链接：https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string

#### 解法：栈+双指针

最开始完全想偏了，想到判断回文字符串上去了，但是那样完全是把题目想复杂了，实现起来比较繁琐且时间效率低，时间复杂度最高可达到O（n^2^）。可能是栈的题目写的少了，最近总是和回文字符串打交道，直接想到那上面去了。

实际上应该是使用一个栈，如果当前字符和栈顶的字符相同，就要弹栈，效率只有O（n）。

这里用数组来实现栈。

```java
class Solution {
    public String removeDuplicates(String S) {
        char[] ch = S.toCharArray();
        int N = ch.length;
        char[] stack = new char[N];
        int top = -1;

        for (int i = 0; i < N; i++) {
            if (top != -1 && stack[top] == ch[i]) {
                top--;
            } else {
                stack[++top] = ch[i];
            }
        }
        String str = new String(stack, 0, top + 1);
        return str;        
    }
}
```

### 1049. 最后一块石头的重量 II

有一堆石头，用整数数组 `stones` 表示。其中 `stones[i]` 表示第 `i` 块石头的重量。每一回合，从中选出**任意两块石头**，然后将它们一起粉碎。假设石头的重量分别为 `x` 和 `y`，且 `x <= y`。那么粉碎的可能结果如下：

- 如果 `x == y`，那么两块石头都会被完全粉碎；
- 如果 `x != y`，那么重量为 `x` 的石头将会完全粉碎，而重量为 `y` 的石头新重量为 `y-x`。

最后，**最多只会剩下一块** 石头。返回此石头 **最小的可能重量** 。如果没有石头剩下，就返回 `0`。

**示例 1：**

```
输入：stones = [2,7,4,1,8,1]
输出：1
解释：
组合 2 和 4，得到 2，所以数组转化为 [2,7,1,8,1]，
组合 7 和 8，得到 1，所以数组转化为 [2,1,1,1]，
组合 2 和 1，得到 1，所以数组转化为 [1,1,1]，
组合 1 和 1，得到 0，所以数组转化为 [1]，这就是最优值。
```

**示例 2：**

```
输入：stones = [31,26,33,21,40]
输出：5
```

**示例 3：**

```
输入：stones = [1,2]
输出：1
```

链接：https://leetcode-cn.com/problems/last-stone-weight-ii

#### 解法：动态规划

这题可以转化为0-1背包问题，问不超过`sum/2`的最大背包重量。`dp[i][j]`表示从物品`stones[0...i]`中选，质量总和不超过`j`的最大重量。

先用二维的解法：

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int len = stones.length;
        int sum = 0;
        for(int i = 0; i < len; i++) {
            sum += stones[i];
        }
        int[][] dp = new int[len+1][sum/2+1];
        //外层是物品
        for(int i = 1; i <= len; i++) {
            int value = stones[i-1];
            for(int j = 1; j <= sum/2; j++) {
                if(j >= value) {
                    //可以选择往背包里加入这个物品，或者不加入这个物品
                    dp[i][j] = Math.max(dp[i-1][j],dp[i-1][j-value]+value);
                }else {
                    //无法把这个物品加入背包
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return sum-2*dp[len][sum/2];
    }
}
```

可以优化空间只用一维数组。内层循环倒序。

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int len = stones.length;
        if(len == 1) return stones[0];
        int sum = 0;
        for(int i = 0; i < len; i++) {
            sum += stones[i];
        }
        int[] dp = new int[sum/2 + 1];
        for(int i = 0; i < len; i++) {
            int value = stones[i];
            for(int j = sum/2; j >= value; j--) {
                dp[j] = Math.max(dp[j],dp[j-value] + value);
            }
        }
        return (sum - 2*dp[sum/2]);
    }
}
```












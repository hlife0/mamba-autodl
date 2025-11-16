#!/usr/bin/env python3
"""
HotpotQA 数据集迭代器
提供便捷的数据访问接口
"""

import json
import random
from typing import List, Tuple, Dict, Any, Iterator, Optional
from dataclasses import dataclass


@dataclass
class HotpotDocument:
    """表示一个上下文文档"""
    title: str
    sentences: List[str]
    
    def get_sentence(self, idx: int) -> Optional[str]:
        """获取指定索引的句子"""
        if 0 <= idx < len(self.sentences):
            return self.sentences[idx]
        return None
    
    def get_full_text(self) -> str:
        """获取文档的完整文本"""
        return " ".join(self.sentences)
    
    def __len__(self) -> int:
        """返回文档中句子的数量"""
        return len(self.sentences)
    
    def __repr__(self) -> str:
        return f"HotpotDocument(title='{self.title}', sentences={len(self.sentences)})"


@dataclass
class HotpotSupportingFact:
    """表示一个支持事实"""
    title: str
    sentence_idx: int
    
    def __repr__(self) -> str:
        return f"SupportingFact('{self.title}', sent_idx={self.sentence_idx})"


@dataclass
class HotpotQAItem:
    """表示一个 HotpotQA 数据条目"""
    id: str
    question: str
    answer: str
    type: str  # 'comparison' or 'bridge'
    level: str  # 'easy', 'medium', or 'hard'
    supporting_facts: List[HotpotSupportingFact]
    context: List[HotpotDocument]
    
    def get_supporting_sentences(self) -> List[Tuple[str, str]]:
        """
        获取所有支持事实对应的句子
        
        Returns:
            List of (title, sentence) tuples
        """
        result = []
        for fact in self.supporting_facts:
            # 在 context 中找到对应的文档
            for doc in self.context:
                if doc.title == fact.title:
                    sentence = doc.get_sentence(fact.sentence_idx)
                    if sentence:
                        result.append((fact.title, sentence))
                    break
        return result
    
    def get_document_by_title(self, title: str) -> Optional[HotpotDocument]:
        """根据标题获取文档"""
        for doc in self.context:
            if doc.title == title:
                return doc
        return None
    
    def get_all_context_text(self) -> str:
        """获取所有上下文的完整文本"""
        texts = []
        for doc in self.context:
            texts.append(f"[{doc.title}] {doc.get_full_text()}")
        return "\n\n".join(texts)
    
    def get_useful(self) -> List[Dict[str, str]]:
        """
        提取有用的文档（supporting_facts中提到的文档）
        
        Returns:
            List of dictionaries, each containing:
                - 'title': 文档标题
                - 'content': 文档完整内容
        """
        result = []
        # 获取所有支持事实中提到的文档标题
        supporting_titles = set(fact.title for fact in self.supporting_facts)
        
        # 从context中提取这些文档
        for doc in self.context:
            if doc.title in supporting_titles:
                result.append({
                    'title': doc.title,
                    'content': doc.get_full_text()
                })
        
        return result
    
    def get_all(self) -> List[Dict[str, str]]:
        """
        提取所有文档
        
        Returns:
            List of dictionaries, each containing:
                - 'title': 文档标题
                - 'content': 文档完整内容
        """
        result = []
        for doc in self.context:
            result.append({
                'title': doc.title,
                'content': doc.get_full_text()
            })
        return result
    
    def __repr__(self) -> str:
        return (f"HotpotQAItem(id='{self.id}', type='{self.type}', level='{self.level}', "
                f"question='{self.question[:50]}...', answer='{self.answer}')")


class HotpotQAIterator:
    """
    HotpotQA 数据集迭代器
    
    用法:
        dataset = HotpotQAIterator("path/to/hotpot.json")
        for item in dataset:
            print(item.question)
            print(item.answer)
    """
    
    def __init__(self, json_path: Optional[str] = None, data: Optional[List[Dict[str, Any]]] = None):
        """
        初始化迭代器
        
        Args:
            json_path: HotpotQA JSON 文件路径（与data二选一）
            data: 直接提供的数据列表（与json_path二选一）
        """
        if json_path is None and data is None:
            raise ValueError("必须提供 json_path 或 data 其中之一")
        
        self.json_path = json_path
        self.data = data
        self._index = 0
        
        if json_path is not None:
            self._load_data()
    
    def _load_data(self):
        """加载 JSON 数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def _parse_item(self, raw_item: Dict[str, Any]) -> HotpotQAItem:
        """
        解析原始数据条目为 HotpotQAItem 对象
        
        Args:
            raw_item: 原始 JSON 数据条目
            
        Returns:
            HotpotQAItem 对象
        """
        # 解析上下文文档
        context = []
        for doc_data in raw_item['context']:
            doc = HotpotDocument(
                title=doc_data[0],
                sentences=doc_data[1]
            )
            context.append(doc)
        
        # 解析支持事实
        supporting_facts = []
        for fact_data in raw_item['supporting_facts']:
            fact = HotpotSupportingFact(
                title=fact_data[0],
                sentence_idx=fact_data[1]
            )
            supporting_facts.append(fact)
        
        # 创建 HotpotQAItem 对象
        item = HotpotQAItem(
            id=raw_item['_id'],
            question=raw_item['question'],
            answer=raw_item['answer'],
            type=raw_item['type'],
            level=raw_item['level'],
            supporting_facts=supporting_facts,
            context=context
        )
        
        return item
    
    def __iter__(self) -> Iterator[HotpotQAItem]:
        """返回迭代器自身"""
        self._index = 0
        return self
    
    def __next__(self) -> HotpotQAItem:
        """返回下一个数据条目"""
        if self._index >= len(self.data):
            raise StopIteration
        
        raw_item = self.data[self._index]
        self._index += 1
        
        return self._parse_item(raw_item)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> HotpotQAItem:
        """
        通过索引访问数据条目

        Args:
            idx: 索引值

        Returns:
            HotpotQAItem 对象
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data)})")

        return self._parse_item(self.data[idx])

    def get_by_id(self, item_id: str) -> Optional[HotpotQAItem]:
        """
        通过ID查找数据条目

        Args:
            item_id: 问题ID

        Returns:
            HotpotQAItem 对象，如果未找到则返回 None
        """
        for raw_item in self.data:
            if raw_item['_id'] == item_id:
                return self._parse_item(raw_item)
        return None
    
    def get_by_type(self, question_type: str) -> 'HotpotQAIterator':
        """
        获取指定类型的所有问题
        
        Args:
            question_type: 'comparison' or 'bridge'
            
        Returns:
            包含符合条件问题的新 HotpotQAIterator
        """
        filtered_data = []
        for raw_item in self.data:
            if raw_item['type'] == question_type:
                filtered_data.append(raw_item)
        return HotpotQAIterator(data=filtered_data)
    
    def get_by_level(self, level: str) -> 'HotpotQAIterator':
        """
        获取指定难度级别的所有问题
        
        Args:
            level: 'easy', 'medium', or 'hard'
            
        Returns:
            包含符合条件问题的新 HotpotQAIterator
        """
        filtered_data = []
        for raw_item in self.data:
            if raw_item['level'] == level:
                filtered_data.append(raw_item)
        return HotpotQAIterator(data=filtered_data)
    
    def random_choose(self, n: int, seed: Optional[int] = None) -> 'HotpotQAIterator':
        """
        从当前迭代器中随机选择 n 个问题
        
        Args:
            n: 要选择的问题数量
            seed: 随机种子（可选，用于可重复的随机选择）
            
        Returns:
            包含随机选择问题的新 HotpotQAIterator
        """
        if n > len(self.data):
            n = len(self.data)
        
        if seed is not None:
            random.seed(seed)
        
        selected_data = random.sample(self.data, n)
        return HotpotQAIterator(data=selected_data)



if __name__ == "__main__":
    """测试 HotpotQA 迭代器"""
    
    print("="*80)
    print("HotpotQA 迭代器测试")
    print("="*80)
    
    # 数据文件路径
    data_path = "/home/hlife/Mamba-analysis/mount/data/HotpotQA/hotpot_train_v1.1.json"
    
    # 创建迭代器
    print(f"\n正在加载数据: {data_path}")
    dataset = HotpotQAIterator(data_path)
    print(f"数据集大小: {len(dataset):,} 条")
    
    # 测试 1: 使用迭代器访问前3个条目
    print("\n" + "="*80)
    print("测试 1: 使用迭代器访问前 3 个条目")
    print("="*80)
    
    for i, item in enumerate(dataset):
        if i >= 3:
            break
        
        print(f"\n【条目 {i+1}】")
        print(f"ID: {item.id}")
        print(f"类型: {item.type}")
        print(f"难度: {item.level}")
        print(f"问题: {item.question}")
        print(f"答案: {item.answer}")
        print(f"上下文文档数: {len(item.context)}")
        print(f"支持事实数: {len(item.supporting_facts)}")
        
        print(f"\n支持事实:")
        for j, fact in enumerate(item.supporting_facts):
            print(f"  {j+1}. {fact}")
        
        print(f"\n支持句子:")
        for j, (title, sentence) in enumerate(item.get_supporting_sentences()):
            print(f"  {j+1}. [{title}]: {sentence}")
    
    # 测试 2: 使用索引访问
    print("\n" + "="*80)
    print("测试 2: 使用索引访问第 100 个条目")
    print("="*80)

    item = dataset[99]  # 第100个条目（索引从0开始）

    print(f"第100个条目的ID: {item.id}")

    # 测试 3: 使用ID访问
    print("\n" + "="*80)
    print("测试 3: 使用ID访问条目")
    print("="*80)

    # 获取第一个条目的ID，然后用ID查找
    first_item_id = dataset[0].id
    print(f"第一个条目的ID: {first_item_id}")

    item_by_id = dataset.get_by_id(first_item_id)
    if item_by_id:
        print(f"✓ 通过ID找到条目:")
        print(f"  问题: {item_by_id.question}")
        print(f"  答案: {item_by_id.answer}")
        print(f"  类型: {item_by_id.type}")
        print(f"  难度: {item_by_id.level}")
    else:
        print("✗ 通过ID未找到条目")

    # 测试不存在的ID
    print(f"\n测试不存在的ID:")
    nonexistent_item = dataset.get_by_id("nonexistent_id_12345")
    if nonexistent_item:
        print("意外找到了条目")
    else:
        print("✓ 正确返回 None (ID不存在)")

    # 测试 4: 获取特定文档
    print("\n" + "="*80)
    print("测试 4: 获取特定文档内容")
    print("="*80)

    item = dataset[0]
    print(f"答案: {item.answer}")
    
    # 测试 3: 获取特定文档
    print("\n" + "="*80)
    print("测试 3: 获取特定文档内容")
    print("="*80)
    
    item = dataset[0]
    doc = item.get_document_by_title("Arthur's Magazine")
    if doc:
        print(f"\n文档标题: {doc.title}")
        print(f"句子数: {len(doc)}")
        print(f"完整文本: {doc.get_full_text()}")
    
    # 测试 4: 按类型过滤（返回Iterator）
    print("\n" + "="*80)
    print("测试 4: 按类型过滤（返回 HotpotQAIterator）")
    print("="*80)
    
    comparison_iter = dataset.get_by_type('comparison')
    bridge_iter = dataset.get_by_type('bridge')
    
    print(f"\nComparison 问题数: {len(comparison_iter):,}")
    print(f"Bridge 问题数: {len(bridge_iter):,}")
    print(f"类型验证: comparison_iter 是 HotpotQAIterator? {isinstance(comparison_iter, HotpotQAIterator)}")
    
    print("\nComparison 类型示例（通过迭代访问）:")
    for i, item in enumerate(comparison_iter):
        if i >= 2:
            break
        print(f"  {i+1}. {item.question}")
    
    print("\nBridge 类型示例（通过索引访问）:")
    for i in range(min(2, len(bridge_iter))):
        item = bridge_iter[i]
        print(f"  {i+1}. {item.question}")
    
    # 测试 5: 按难度过滤（返回Iterator）
    print("\n" + "="*80)
    print("测试 5: 按难度过滤（返回 HotpotQAIterator）")
    print("="*80)
    
    easy_iter = dataset.get_by_level('easy')
    medium_iter = dataset.get_by_level('medium')
    hard_iter = dataset.get_by_level('hard')
    
    print(f"\nEasy 问题数: {len(easy_iter):,}")
    print(f"Medium 问题数: {len(medium_iter):,}")
    print(f"Hard 问题数: {len(hard_iter):,}")
    print(f"类型验证: easy_iter 是 HotpotQAIterator? {isinstance(easy_iter, HotpotQAIterator)}")
    
    # 测试 6: random_choose 功能
    print("\n" + "="*80)
    print("测试 6: random_choose 随机选择")
    print("="*80)
    
    # 从整个数据集随机选10个
    random_10 = dataset.random_choose(10, seed=42)
    print(f"\n从整个数据集随机选择 10 个问题:")
    print(f"选择结果大小: {len(random_10)}")
    print(f"类型验证: random_10 是 HotpotQAIterator? {isinstance(random_10, HotpotQAIterator)}")
    
    print("\n随机选择的问题:")
    for i, item in enumerate(random_10):
        print(f"  {i+1}. [{item.type}] {item.question[:60]}...")
    
    # 测试 7: 链式调用
    print("\n" + "="*80)
    print("测试 7: 链式调用（过滤+随机选择）")
    print("="*80)
    
    # 先按类型过滤，再随机选择
    comparison_random_5 = dataset.get_by_type('comparison').random_choose(5, seed=123)
    print(f"\n从 comparison 类型中随机选择 5 个:")
    print(f"选择结果大小: {len(comparison_random_5)}")
    
    for i, item in enumerate(comparison_random_5):
        print(f"  {i+1}. {item.question[:60]}...")
    
    # 多重链式调用：类型过滤 + 难度过滤 + 随机选择
    print("\n多重链式调用: bridge -> easy -> 随机选3个")
    bridge_easy = dataset.get_by_type('bridge').get_by_level('easy')
    print(f"Bridge + Easy 总数: {len(bridge_easy)}")
    
    if len(bridge_easy) > 0:
        bridge_easy_random = bridge_easy.random_choose(min(3, len(bridge_easy)), seed=456)
        print(f"随机选择: {len(bridge_easy_random)} 个")
        for i, item in enumerate(bridge_easy_random):
            print(f"  {i+1}. {item.question[:60]}...")
    
    # 测试 8: 验证迭代器可以重复迭代
    print("\n" + "="*80)
    print("测试 8: 验证迭代器可以重复迭代")
    print("="*80)
    
    small_iter = dataset.random_choose(3, seed=999)
    
    print("\n第一次迭代:")
    for i, item in enumerate(small_iter):
        print(f"  {i+1}. {item.question[:40]}...")
    
    print("\n第二次迭代（应该得到相同结果）:")
    for i, item in enumerate(small_iter):
        print(f"  {i+1}. {item.question[:40]}...")
    
    # 测试 9: get_useful() 和 get_all() 方法
    print("\n" + "="*80)
    print("测试 9: get_useful() 和 get_all() 方法")
    print("="*80)
    
    item = dataset[0]
    
    print(f"\n问题: {item.question}")
    print(f"答案: {item.answer}")
    
    # 测试 get_useful()
    print("\n--- get_useful() 提取有用文档 ---")
    useful_docs = item.get_useful()
    print(f"有用文档数: {len(useful_docs)}")
    print(f"返回类型: {type(useful_docs)}")
    print(f"第一个元素类型: {type(useful_docs[0]) if useful_docs else 'N/A'}")
    
    for i, doc_dict in enumerate(useful_docs):
        print(f"\n文档 {i+1}:")
        print(f"  标题: {doc_dict['title']}")
        print(f"  内容长度: {len(doc_dict['content'])} 字符")
        print(f"  内容预览: {doc_dict['content'][:100]}...")
    
    # 测试 get_all()
    print("\n--- get_all() 提取所有文档 ---")
    all_docs = item.get_all()
    print(f"所有文档数: {len(all_docs)}")
    print(f"返回类型: {type(all_docs)}")
    
    print("\n所有文档标题列表:")
    for i, doc_dict in enumerate(all_docs):
        # 检查是否是有用文档
        is_useful = doc_dict['title'] in [d['title'] for d in useful_docs]
        marker = " ✅ 有用" if is_useful else " (干扰项)"
        print(f"  {i+1}. {doc_dict['title']}{marker}")
    
    # 验证结构
    print("\n验证返回格式:")
    if all_docs:
        sample_doc = all_docs[0]
        print(f"  字典键: {list(sample_doc.keys())}")
        print(f"  'title' 类型: {type(sample_doc['title'])}")
        print(f"  'content' 类型: {type(sample_doc['content'])}")
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)


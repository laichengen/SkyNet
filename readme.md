# 天网（SkyNet）
这是一个大模型的项目，同时利用了RAG，Agent和RLHF技术，如果你想从事这方面的工作，该项目可以很好的帮助你系统性地理解相关知识
（This is a large language model project that uses RAG, Agent, and RLHF to help learners understand relevant knowledge. ）

# 天网具备的能力（The ability of SkyNet）

1. 天网支持在数学等逻辑问题上可以给出一个准确的答案（SkyNet supports providing accurate answers to logical problems such as mathematics）
2. 天网支持利用本地数据资源或者网络资源进行自主学习（SkyNet supports autonomous learning using local data resources or network resources）
3. 天网支持对自身缺陷进行自我纠正（Skynet supports self correction of its own defects）

# 天网使用的技术（The technology utilized by Skynet）

1. Agent接口--将自然语言转成代码，然后进行代码层面上的运行得到结果（Agent Interface - Convert natural language into code and run it at the code level to obtain results）
2. Agent接口--可以进行谷歌搜索api的调用，进行网上资源的爬取（Agent interface - Call Google search API and crawl online resources）
3. Rag技术--将爬取的网上资源作为现阶段的知识，LLM基于知识进行总结学习（Rag technology - using crawled online resources as current knowledge, LLM summarizes and learns based on knowledge）
4. RLHF技术--将运行失败的代码作为负样本，运行成功的代码作为正样本进行Memory，然后通过RLHF对模型进行自我纠正（RLHF technology - using failed code as negative samples and successful code as positive samples for memory, and then self correcting the model through RLHF）

# 参考代码（The Reference）
1. https://github.com/wdndev/llm_interview_note
2. https://github.com/datawhalechina/tiny-universe 
3. https://github.com/ethanyanjiali/minChatGPT

# 训练（Training）
1. 安装依赖（Install dependencies with）
```bash
pip install -r requirements.txt
```
2. 下载模型（Download models）
```bash
Download bert-base-uncased model from huggingface website to dir 'checkpoint/Embeddings-bert-base-uncased'
Download GPT2 model from huggingface website to dir 'checkpoint/GPT2'
Download Qwen2-Coder-0.5B-Instruct model from huggingface website to dir 'checkpoint/Qwen2-Coder-0.5B-Instruct'
```
3. 运行代码（Run the code）
```bash
python main.py
```
# 界面语言包 / Interface language packs

在左侧 Language / 语言 中选择“简体中文”或“English”。界面语言覆盖导航、参数、帮助、提示、表格控件和图片导出。数据中的样本名、分组名、注释名称及字段值按原始内容显示。

内置语言包位于 `scMM/locales/zh_CN.json` 和 `scMM/locales/en.json`，运行时从本地加载。可复制一个语言包，修改 `code`、`name` 和 `messages` 中的译文，再将文件放入该目录。重启程序后，新语言会出现在语言列表中。

也可设置环境变量 `SCMM_LANGUAGE_DIR` 指向附加语言包目录。相同 `code` 的外部语言包优先加载。

```json
{
  "code": "fr",
  "name": "Français",
  "messages": {
    "图片导出": "Exporter la figure",
    "选定细胞：{0} / {1}": "Cellules sélectionnées : {0} / {1}"
  }
}
```

`messages` 左侧是界面原文，右侧是译文。翻译时保留 `{0}`、`{1}` 等占位符及 Markdown 标记，可调整占位符顺序。未提供译文的文字显示原文。参数默认值继续在 `scMM/config/defaults.json` 中配置，语言包负责显示文字。

Select Simplified Chinese or English under **Language / 语言** in the sidebar. Interface text is loaded locally. Sample names, group names, compound annotations and stored values retain their original content.

To add a language, copy an existing JSON pack from `scMM/locales`, set a unique `code` and display `name`, and translate the values in `messages`. Preserve the source keys, numbered placeholders and Markdown formatting. Put the pack in `scMM/locales` or in the directory specified by `SCMM_LANGUAGE_DIR`, then restart SCMM. External packs take precedence for matching language codes. Missing translations use the source text.

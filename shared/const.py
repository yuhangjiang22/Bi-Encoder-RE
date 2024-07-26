task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'chemprot': ['CHEMICAL', 'GENE'],
    'chemprot_5': ['CHEMICAL', 'GENE'],
    'biored': ['DiseaseOrPhenotypicFeature', 'SequenceVariant', 'GeneOrGeneProduct', 'ChemicalEntity'],
    'semeval': ['obj', 'sub'],
    'ddi': ["DRUG", "BRAND", "GROUP", "DRUG_N"]
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'chemprot': ['UPREGULATOR', 'ACTIVATOR', 'INDIRECT-UPREGULATOR',
                 'DOWNREGULATOR', 'INHIBITOR', 'INDIRECT-DOWNREGULATOR',
                 'AGONIST', 'AGONIST-ACTIVATOR', 'AGONIST-INHIBITOR',
                 'ANTAGONIST',
                 'SUBSTRATE', 'PRODUCT-OF', 'SUBSTRATE_PRODUCT-OF'],
    'chemprot_5': ['CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9'],
    'biored': ['Positive_Correlation', 'Negative_Correlation', 'Association', 'Bind', 'Drug_Interaction', 'Cotreatment', 'Comparison', 'Conversion'],
    'semeval': ['Cause-Effect', 'Instrument-Agency', 'Product-Producer', 'Content-Container', 'Entity-Origin', 'Entity-Destination',
                'Component-Whole', 'Member-Collection', 'Message-Topic'],
    'ddi': ['EFFECT', 'INT', 'MECHANISM', 'ADVISE']
}

# task_id2descriptions = {
#     'chemprot_5': {0: ["there are no relations between the compound @subject@ and gene @object@ .",
#                       "the compound @subject@ and gene @object@ has no relations ."],
#                   1: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
#                       "upregulator , activator , or indirect upregulator in its interactions .",
#                       "@subject@ initiates or enhances the activity of @object@ through direct or indirect means . an "
#                       "upregulator ,activator , or indirect upregulator serves as the mechanism that increases the "
#                       "function ,"
#                       "expression , or activity of the @object@"
#                       ],
#                   2: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
#                       "downregulator , inhibitor , or indirect downregulator in its interactions .",
#                       "@subject@ interacts with the gene @object@ , resulting in a decrease in the gene's "
#                       "activity or expression . This interaction can occur through direct inhibition , acting as a "
#                       "downregulator , or through indirect means , where the compound causes a reduction in the gene's "
#                       "function or expression without directly binding to it . Such mechanisms are crucial in "
#                       "understanding genetic regulation and can have significant implications in fields like "
#                       "pharmacology and gene therapy ."
#                       ],
#                   3: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
#                       "agonist , agonist activator , or agonist inhibitor in its interactions .",
#                       "@subject@ interacts with the gene @object@ in a manner that modulates its activity positively ( "
#                       "as an agonist or agonist activator ) or negatively ( as an agonist inhibitor ) . An agonist "
#                       "interaction typically increases the gene's activity or the activity of proteins expressed by "
#                       "the gene , whereas an agonist activator enhances this effect further . Conversely , an agonist "
#                       "inhibitor would paradoxically bind in a manner that initially mimics an agonist's action but "
#                       "ultimately inhibits the gene's activity or its downstream effects ."
#                       ],
#                   4: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
#                       "antagonist in its interactions .",
#                       "@subject@ interacts with the gene @object@ by acting as an antagonist . This means that the "
#                       "compound blocks or diminishes the gene's normal activity or the activity of the protein product "
#                       "expressed by the gene . Antagonist interactions are significant in the regulation of biological "
#                       "pathways and have wide-ranging implications in therapeutic interventions , where they can be "
#                       "used to modulate the effects of genes involved in disease processes ."
#                       ],
#                   5: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
#                       "substrate , product of, or substrate product of in its interactions .",
#                       "@subject@ engages with the gene @object@ in a manner where it acts as a substrate , is a product"
#                       "of, or both a substrate and product within the gene's associated biochemical pathways ."
#                       ]},
#     'chemprot': {0: [" there are no relations between the compound @subject@ and gene @object@ .",],
#                   1: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
#                       "upregulator in its interactions . ",
#                       ],
#                   2: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an activator in its interactions . "],
#                   3: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an indirect upregulator in its interactions . "],
#
#                   4: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
#                       "downregulator in its interactions .",
#                       ],
#                   5: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an inhibitor in its interactions . "],
#                   6: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an indirect downregulator in its interactions . "],
#
#                   7: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
#                       "agonist in its interactions .",
#                       ],
#                   8: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an agonist activator in its interactions . "],
#                   9: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an agonist inhibitor in its interactions . "],
#
#                   10: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
#                       "antagonist in its interactions .",
#                       ],
#
#                   11: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
#                       "substrate in its interactions .",
#                       ],
#                   12: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a product of in its interactions . "],
#                   13: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a substrate product of in its interactions . "],},
#

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label

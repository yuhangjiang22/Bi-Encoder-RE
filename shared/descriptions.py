descriptions = {
    'scierc': {0: ["no relation : there are no relations between the @subject@ and the @object@ ."],
                  1: ["part of : the @subject@ is a component or segment that is integral to the structure or composition "
                      "of the @object@ ."],
                  2: ["used for : the @subject@ is a tool or method applied to enhance or facilitate the @object@ ."],
                  3: ["feature of : the @subject@ is a constituent part or characteristic of the @object@ , functioning as a "
                      "distinctive element within the @object@ , or falls within the scope or area of expertise defined "
                      "by the domain of the @object@ ."],
                  4: ["conjunction : the @subject@ serves a role or purpose analogous to the @object@ , often being used in "
                      "conjunction with or incorporated into the @object@ for complementary or similar functions . "],
                  5: ["evaluate for : the @object@ is assessed or analyzed specifically to determine its suitability , "
                      "effectiveness , or performance in relation to the @subject@ ."],
                  6: ["hyponym of : the @subject@ is a specific instance or category under the broader classification of "
                      "@object@ , signifying that the @subject@ is a subtype or a more specialized form within the general "
                      "framework of the @object@ ."],
                  7: ["compare : the @subject@ is compared in relation to the @object@ , highlighting similarities and "
                      "differences to understand their respective characteristics or performances ."]},
    'chemprot': {0: [" there are no relations between the compound @subject@ and gene @object@ .",],
                  1: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
                      "upregulator in its interactions . ",
                      ],
                  2: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an activator in its interactions . "],
                  3: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an indirect upregulator in its interactions . "],

                  4: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
                      "downregulator in its interactions .",
                      ],
                  5: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an inhibitor in its interactions . "],
                  6: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an indirect downregulator in its interactions . "],

                  7: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
                      "agonist in its interactions .",
                      ],
                  8: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an agonist activator in its interactions . "],
                  9: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an agonist inhibitor in its interactions . "],

                  10: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
                      "antagonist in its interactions .",
                      ],

                  11: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
                      "substrate in its interactions .",
                      ],
                  12: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a product of in its interactions . "],
                  13: [" the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a substrate product of in its interactions . "],},
    'chemprot_5': {0: ["there are no relations between the compound @subject@ and gene @object@ .",
                      "the compound @subject@ and gene @object@ has no relations ."],
                  1: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
                      "upregulator , activator , or indirect upregulator in its interactions .",
                      "@subject@ initiates or enhances the activity of @object@ through direct or indirect means . an "
                      "upregulator ,activator , or indirect upregulator serves as the mechanism that increases the "
                      "function ,"
                      "expression , or activity of the @object@"
                      ],
                  2: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
                      "downregulator , inhibitor , or indirect downregulator in its interactions .",
                      "@subject@ interacts with the gene @object@ , resulting in a decrease in the gene's "
                      "activity or expression . This interaction can occur through direct inhibition , acting as a "
                      "downregulator , or through indirect means , where the compound causes a reduction in the gene's "
                      "function or expression without directly binding to it . Such mechanisms are crucial in "
                      "understanding genetic regulation and can have significant implications in fields like "
                      "pharmacology and gene therapy ."
                      ],
                  3: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
                      "agonist , agonist activator , or agonist inhibitor in its interactions .",
                      "@subject@ interacts with the gene @object@ in a manner that modulates its activity positively ( "
                      "as an agonist or agonist activator ) or negatively ( as an agonist inhibitor ) . An agonist "
                      "interaction typically increases the gene's activity or the activity of proteins expressed by "
                      "the gene , whereas an agonist activator enhances this effect further . Conversely , an agonist "
                      "inhibitor would paradoxically bind in a manner that initially mimics an agonist's action but "
                      "ultimately inhibits the gene's activity or its downstream effects ."
                      ],
                  4: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
                      "antagonist in its interactions .",
                      "@subject@ interacts with the gene @object@ by acting as an antagonist . This means that the "
                      "compound blocks or diminishes the gene's normal activity or the activity of the protein product "
                      "expressed by the gene . Antagonist interactions are significant in the regulation of biological "
                      "pathways and have wide-ranging implications in therapeutic interventions , where they can be "
                      "used to modulate the effects of genes involved in disease processes ."
                      ],
                  5: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
                      "substrate , product of, or substrate product of in its interactions .",
                      "@subject@ engages with the gene @object@ in a manner where it acts as a substrate , is a product"
                      "of, or both a substrate and product within the gene's associated biochemical pathways ."
                      ]},
    'biored': {
    'ChemicalEntity-DiseaseOrPhenotypicFeature': {
        0: ['there are no relations between the drug @ChemicalEntity@ and disease @DiseaseOrPhenotypicFeature@ .',],
        1: ['the drug @ChemicalEntity@ may induce the disease @DiseaseOrPhenotypicFeature@ , increase its risk, '
            'or the levels may'
            'correlate with disease risk .'],
        2: ['the drug @ChemicalEntity@ is able to treat the disease @DiseaseOrPhenotypicFeature@ or decrease its '
            'susceptibility .'],
        3: ['the drug @ChemicalEntity@ is found to affect the disease @DiseaseOrPhenotypicFeature@ but not clearly '
            'identified as'
            'positive or negative correlations . '],
        4: [''],
        5: [''],
        6: [''],
        7: [''],
        8: [''],
    },
    'GeneOrGeneProduct-DiseaseOrPhenotypicFeature': {
        0: ['there are no relations between the disease @DiseaseOrPhenotypicFeature@ and gene @GeneOrGeneProduct@ .',],
        1: ['overexpression or side effects of the gene @GeneOrGeneProduct@ may cause the disease '
            '@DiseaseOrPhenotypicFeature@ .'],
        2: ['the proteins from the gene @GeneOrGeneProduct@ used as drugs may treat the disease '
            '@DiseaseOrPhenotypicFeature@ or the absence may'
            'cause diseases .'],
        3: ['the functional gene @GeneOrGeneProduct@ prevents disease @DiseaseOrPhenotypicFeature@ or other '
            'association relationships .' ],
        4: [''],
        5: [''],
        6: [''],
        7: [''],
        8: [''],
    },
    'SequenceVariant-DiseaseOrPhenotypicFeature': {
        0: ['there are no relations between the disease @DiseaseOrPhenotypicFeature@ and variant @SequenceVariant@ .',],
        1: ['the variant @SequenceVariant@ may increase disease @DiseaseOrPhenotypicFeature@ risk, contribute to '
            'disease susceptibility,'
            'or cause protein deficiencies leading to diseases .'],
        2: ['the variant @SequenceVariant@ may increase disease @DiseaseOrPhenotypicFeature@ risk .'],
        3: ['the variant @SequenceVariant@ associated with the disease @DiseaseOrPhenotypicFeature@ prevalence and '
            'which that cannot be'
            'categorized as causing the disease . '],
        4: [''],
        5: [''],
        6: [''],
        7: [''],
        8: [''],
    },
    'GeneOrGeneProduct-GeneOrGeneProduct': {
        0: ['there are no relations between the gene @GeneOrGeneProduct@ and gene @GeneOrGeneProduct@ .'],
        1: ['the gene @GeneOrGeneProduct@ and gene @GeneOrGeneProduct@ may show positive correlations in expression '
            'or regulatory functions .'],
        2: ['the gene @GeneOrGeneProduct@ and gene @GeneOrGeneProduct@ may show negative correlations in expression '
            'or regulatory functions .'],
        3: ['associations between gene @GeneOrGeneProduct@ and gene @GeneOrGeneProduct@ that cannot be categorized '
            'differently .'],
        4: ['there are physical interactions between proteins from gene @GeneOrGeneProduct@ and gene '
            '@GeneOrGeneProduct@ , including protein'
            'binding at gene promoters .'],
        5: [''],
        6: [''],
        7: [''],
        8: [''],
    },
    'GeneOrGeneProduct-ChemicalEntity': {
        0: ['there are no relations between the drug @ChemicalEntity@ and gene @GeneOrGeneProduct@ .'],
        1: ['the drug @ChemicalEntity@ may cause higher expression of gene @GeneOrGeneProduct@ or gene variants may '
            'trigger'
            'chemical adverse effects .'],
        2: ['the drug @ChemicalEntity@ may cause lower expression of gene @GeneOrGeneProduct@ or gene variants may '
            'confer'
            'resistance to chemicals .'],
        3: ['there are non-specific associations and binding interactions between the drug @ChemicalEntity@ and gene '
            '@GeneOrGeneProduct@ promoters .'],
        4: ['there are relations between the gene @GeneOrGeneProduct@ and the drug @ChemicalEntity@ such that the '
            'drug binds the'
            'promoter of a gene, or the protein from the gene is the drug receptor .'],
        5: [''],
        6: [''],
        7: [''],
        8: [''],
    },
    'ChemicalEntity-ChemicalEntity': {
        0: ['there are no relations between the drug @ChemicalEntity@ and drug @ChemicalEntity@ .'],
        1: ['the drug @ChemicalEntity@ may increase the sensitivity or effectiveness of drug @ChemicalEntity@ or vice '
            'versa .'],
        2: ['the drug @ChemicalEntity@ may decrease the sensitivity or side effects of drug @ChemicalEntity@ or vice '
            'versa .'],
        3: ['there are chemical conversions or non-specific associations between drug @ChemicalEntity@ and drug '
            '@ChemicalEntity@ .'],
        4: [''],
        5: ['there are pharmacodynamic interactions between the drug @ChemicalEntity@ and drug @ChemicalEntity@ .'],
        6: ['the drug combination therapy using both drug @ChemicalEntity@ and drug @ChemicalEntity@ .'],
        7: ['there is a comparison relation between drug @ChemicalEntity@ and drug @ChemicalEntity@ .'],
        8: ['the drug @ChemicalEntity@ may convert to drug @ChemicalEntity@ or vice versa .'],
    },
    'ChemicalEntity-SequenceVariant': {
        0: ['there are no relations between the drug @ChemicalEntity@ and variant @SequenceVariant@ .'],
        1: ['the drug @ChemicalEntity@ may cause higher expression of a gene variant @SequenceVariant@ or increase '
            'sensitivity due to a variant .'],
        2: ['the drug @ChemicalEntity@ may decrease gene expression due to the variant @SequenceVariant@ or the '
            'variant may'
            'confer resistance .'],
        3: ['there are association relationships not defined between the variant @SequenceVariant@ and the drug '
            '@ChemicalEntity@ , like variant on chemical binding sites .'],
        4: [''],
        5: [''],
        6: [''],
        7: [''],
        8: [''],

    },
    'SequenceVariant-SequenceVariant': {
        0: ['there are no relations between the variant @SequenceVariant@ and variant @SequenceVariant@ .'],
        1: [''],
        2: [''],
        3: ['there is a association relation between the variant @SequenceVariant@ and variant @SequenceVariant@ .'],
        4: [''],
        5: [''],
        6: [''],
        7: [''],
        8: [''],
    }},



}
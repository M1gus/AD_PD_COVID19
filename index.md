# Alzheimer’s and Parkinson’s diseases predict different COVID-19 outcomes, a UK Biobank study

## Author: Yizhou Yu
### Affiliation: MRC Toxicology, University of Cambridge


Contract detail: yzy21 [at] mrc-tox.cam.ac.uk

Date: November 2020

The input files for this analysis pipeline are on the master branch of this GitHub page: 
https://github.com/M1gus/AD_PD_COVID19

This study was first deposited in a preprint server (medRxiv) under the title "Alzheimer’s and Parkinson’s diseases predict different COVID-19 outcomes, a UK Biobank study" Authors: Yizhou Yu, Marco Travaglio,  Rebeka Popovic, Nuno Santos Leal, L. Miguel Martins



Here, I will first extract the UK Biobank data, from all >500,000 people. I will then subset the data from all people who have been tested for COVID-19.

I will analyse the data of participants who were tested and infected (without information on death). This will allow me to determine what drives infection. <br>
In this analysis, I will try to include: UK Biobank phenotype and genotype markers (e.g. BMI, smoking status, dementia, etc.), the crowd mobility that they were exposed to, etc. 

I will then look at how these markers affect lethality.

Access to the UK Biobank data is granted under the application 60124.

Note that I had to downgrade R to run it with rpy2<br>
update.packages(ask=FALSE,
                checkBuilt=TRUE,
                repos="https://cloud.r-project.org")

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Alzheimer’s-and-Parkinson’s-diseases-predict-different-COVID-19-outcomes,-a-UK-Biobank-study" data-toc-modified-id="Alzheimer’s-and-Parkinson’s-diseases-predict-different-COVID-19-outcomes,-a-UK-Biobank-study-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Alzheimer’s and Parkinson’s diseases predict different COVID-19 outcomes, a UK Biobank study</a></span><ul class="toc-item"><li><span><a href="#Author:-Yizhou-Yu" data-toc-modified-id="Author:-Yizhou-Yu-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Author: Yizhou Yu</a></span><ul class="toc-item"><li><span><a href="#Affiliation:-MRC-Toxicology,-University-of-Cambridge" data-toc-modified-id="Affiliation:-MRC-Toxicology,-University-of-Cambridge-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Affiliation: MRC Toxicology, University of Cambridge</a></span></li></ul></li></ul></li><li><span><a href="#Data-Curation" data-toc-modified-id="Data-Curation-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Curation</a></span><ul class="toc-item"><li><span><a href="#Set-R-environment-&amp;-Load-python-libraries" data-toc-modified-id="Set-R-environment-&amp;-Load-python-libraries-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Set R environment &amp; Load python libraries</a></span></li><li><span><a href="#Obtain-the-UK-Biobank-phenotype-data-subset,-with-the-variables-of-interest." data-toc-modified-id="Obtain-the-UK-Biobank-phenotype-data-subset,-with-the-variables-of-interest.-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Obtain the UK Biobank phenotype data subset, with the variables of interest.</a></span><ul class="toc-item"><li><span><a href="#Create-a-selector-function" data-toc-modified-id="Create-a-selector-function-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Create a selector function</a></span></li><li><span><a href="#Structure-of-the-curated-data" data-toc-modified-id="Structure-of-the-curated-data-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Structure of the curated data</a></span></li></ul></li><li><span><a href="#Subset-the-participants-with-COVID-19-results-(as-of-August-2020)" data-toc-modified-id="Subset-the-participants-with-COVID-19-results-(as-of-August-2020)-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Subset the participants with COVID-19 results (as of August 2020)</a></span><ul class="toc-item"><li><span><a href="#Load-UK-Biobank-COVID-19-data" data-toc-modified-id="Load-UK-Biobank-COVID-19-data-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Load UK Biobank COVID-19 data</a></span><ul class="toc-item"><li><span><a href="#Get-the-date-range-of-the-test-results" data-toc-modified-id="Get-the-date-range-of-the-test-results-2.3.1.1"><span class="toc-item-num">2.3.1.1&nbsp;&nbsp;</span>Get the date range of the test results</a></span></li></ul></li><li><span><a href="#Save-curated-data" data-toc-modified-id="Save-curated-data-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Save curated data</a></span></li></ul></li><li><span><a href="#Mergeing-the-2-parts-of-this-dataset" data-toc-modified-id="Mergeing-the-2-parts-of-this-dataset-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Mergeing the 2 parts of this dataset</a></span><ul class="toc-item"><li><span><a href="#Here,-I-will-get-the-most-up-to-date-result" data-toc-modified-id="Here,-I-will-get-the-most-up-to-date-result-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>Here, I will get the most up-to-date result</a></span></li><li><span><a href="#Subset-the-data-with-only-the-COVID-19-tested-participants" data-toc-modified-id="Subset-the-data-with-only-the-COVID-19-tested-participants-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>Subset the data with only the COVID-19-tested participants</a></span><ul class="toc-item"><li><span><a href="#Select-rows-by-chunk" data-toc-modified-id="Select-rows-by-chunk-2.4.2.1"><span class="toc-item-num">2.4.2.1&nbsp;&nbsp;</span>Select rows by chunk</a></span></li><li><span><a href="#Select-rows-by-chunk" data-toc-modified-id="Select-rows-by-chunk-2.4.2.2"><span class="toc-item-num">2.4.2.2&nbsp;&nbsp;</span>Select rows by chunk</a></span></li></ul></li><li><span><a href="#Get-the-most-up-to-date-data-from-these-participants" data-toc-modified-id="Get-the-most-up-to-date-data-from-these-participants-2.4.3"><span class="toc-item-num">2.4.3&nbsp;&nbsp;</span>Get the most up-to-date data from these participants</a></span></li><li><span><a href="#Decode-with-the-UK-Biobank-values" data-toc-modified-id="Decode-with-the-UK-Biobank-values-2.4.4"><span class="toc-item-num">2.4.4&nbsp;&nbsp;</span>Decode with the UK Biobank values</a></span><ul class="toc-item"><li><span><a href="#Prepare-encodings" data-toc-modified-id="Prepare-encodings-2.4.4.1"><span class="toc-item-num">2.4.4.1&nbsp;&nbsp;</span>Prepare encodings</a></span></li><li><span><a href="#Decode-UKB-phenotype-data" data-toc-modified-id="Decode-UKB-phenotype-data-2.4.4.2"><span class="toc-item-num">2.4.4.2&nbsp;&nbsp;</span>Decode UKB phenotype data</a></span></li><li><span><a href="#Assign-custom-column-names" data-toc-modified-id="Assign-custom-column-names-2.4.4.3"><span class="toc-item-num">2.4.4.3&nbsp;&nbsp;</span>Assign custom column names</a></span></li></ul></li><li><span><a href="#Merge-the-UK-Biobank-phenotype-data-with-the-UK-Biobank-COVID-19-data" data-toc-modified-id="Merge-the-UK-Biobank-phenotype-data-with-the-UK-Biobank-COVID-19-data-2.4.5"><span class="toc-item-num">2.4.5&nbsp;&nbsp;</span>Merge the UK Biobank phenotype data with the UK Biobank COVID-19 data</a></span><ul class="toc-item"><li><span><a href="#Curate-the-PD/AD-diagnosis-data" data-toc-modified-id="Curate-the-PD/AD-diagnosis-data-2.4.5.1"><span class="toc-item-num">2.4.5.1&nbsp;&nbsp;</span>Curate the PD/AD diagnosis data</a></span></li><li><span><a href="#Hypertension-/-high-blood-pressure-calculations" data-toc-modified-id="Hypertension-/-high-blood-pressure-calculations-2.4.5.2"><span class="toc-item-num">2.4.5.2&nbsp;&nbsp;</span>Hypertension / high blood pressure calculations</a></span></li><li><span><a href="#Calculate-Age:-2020---birth-year" data-toc-modified-id="Calculate-Age:-2020---birth-year-2.4.5.3"><span class="toc-item-num">2.4.5.3&nbsp;&nbsp;</span>Calculate Age: 2020 - birth year</a></span></li><li><span><a href="#Drop-the-unused-columns" data-toc-modified-id="Drop-the-unused-columns-2.4.5.4"><span class="toc-item-num">2.4.5.4&nbsp;&nbsp;</span>Drop the unused columns</a></span></li></ul></li><li><span><a href="#Create-Descriptive-table---obsolete" data-toc-modified-id="Create-Descriptive-table---obsolete-2.4.6"><span class="toc-item-num">2.4.6&nbsp;&nbsp;</span>Create Descriptive table - obsolete</a></span></li></ul></li><li><span><a href="#Add-the-exposure-to-air-pollution" data-toc-modified-id="Add-the-exposure-to-air-pollution-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Add the exposure to air pollution</a></span><ul class="toc-item"><li><span><a href="#Convert-UKB-X-&amp;-Y-to-long-lat" data-toc-modified-id="Convert-UKB-X-&amp;-Y-to-long-lat-2.5.1"><span class="toc-item-num">2.5.1&nbsp;&nbsp;</span>Convert UKB X &amp; Y to long lat</a></span></li><li><span><a href="#Graph-the-location-of-the-UK-Biobank-participants" data-toc-modified-id="Graph-the-location-of-the-UK-Biobank-participants-2.5.2"><span class="toc-item-num">2.5.2&nbsp;&nbsp;</span>Graph the location of the UK Biobank participants</a></span></li><li><span><a href="#Match-the-location-of-air-pollution-levels-to-location-of-UK-Biobank-participants" data-toc-modified-id="Match-the-location-of-air-pollution-levels-to-location-of-UK-Biobank-participants-2.5.3"><span class="toc-item-num">2.5.3&nbsp;&nbsp;</span>Match the location of air pollution levels to location of UK Biobank participants</a></span></li><li><span><a href="#Assign-a-population-density-value-to-each-UK-Biobank-participant" data-toc-modified-id="Assign-a-population-density-value-to-each-UK-Biobank-participant-2.5.4"><span class="toc-item-num">2.5.4&nbsp;&nbsp;</span>Assign a population density value to each UK Biobank participant</a></span><ul class="toc-item"><li><span><a href="#Use-the-modelled-N-of-people-per-30m2" data-toc-modified-id="Use-the-modelled-N-of-people-per-30m2-2.5.4.1"><span class="toc-item-num">2.5.4.1&nbsp;&nbsp;</span>Use the modelled N of people per 30m2</a></span></li><li><span><a href="#Investigation-on-movement-data" data-toc-modified-id="Investigation-on-movement-data-2.5.4.2"><span class="toc-item-num">2.5.4.2&nbsp;&nbsp;</span>Investigation on movement data</a></span></li><li><span><a href="#We-will-not-include-the-mobility-data" data-toc-modified-id="We-will-not-include-the-mobility-data-2.5.4.3"><span class="toc-item-num">2.5.4.3&nbsp;&nbsp;</span>We will not include the mobility data</a></span></li></ul></li></ul></li><li><span><a href="#Merge-generated-data-with-data-from-the-UK-Biobank" data-toc-modified-id="Merge-generated-data-with-data-from-the-UK-Biobank-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Merge generated data with data from the UK Biobank</a></span><ul class="toc-item"><li><span><a href="#Drop-unused-columns" data-toc-modified-id="Drop-unused-columns-2.6.1"><span class="toc-item-num">2.6.1&nbsp;&nbsp;</span>Drop unused columns</a></span></li><li><span><a href="#Add-2-more-variables-to-the-list:-education-and-care-home" data-toc-modified-id="Add-2-more-variables-to-the-list:-education-and-care-home-2.6.2"><span class="toc-item-num">2.6.2&nbsp;&nbsp;</span>Add 2 more variables to the list: education and care home</a></span><ul class="toc-item"><li><span><a href="#Get-the-most-up-to-date-data-from-these-participants" data-toc-modified-id="Get-the-most-up-to-date-data-from-these-participants-2.6.2.1"><span class="toc-item-num">2.6.2.1&nbsp;&nbsp;</span>Get the most up-to-date data from these participants</a></span></li><li><span><a href="#Decode" data-toc-modified-id="Decode-2.6.2.2"><span class="toc-item-num">2.6.2.2&nbsp;&nbsp;</span>Decode</a></span></li></ul></li><li><span><a href="#Replace-&quot;No-Answer&quot;-as-NA" data-toc-modified-id="Replace-&quot;No-Answer&quot;-as-NA-2.6.3"><span class="toc-item-num">2.6.3&nbsp;&nbsp;</span>Replace "No Answer" as NA</a></span></li><li><span><a href="#Add-addtional-variables:-blood-type,-C-reactive-proteins-and-image-derived-variable" data-toc-modified-id="Add-addtional-variables:-blood-type,-C-reactive-proteins-and-image-derived-variable-2.6.4"><span class="toc-item-num">2.6.4&nbsp;&nbsp;</span>Add addtional variables: blood type, C-reactive proteins and image derived variable</a></span><ul class="toc-item"><li><span><a href="#Data-curation" data-toc-modified-id="Data-curation-2.6.4.1"><span class="toc-item-num">2.6.4.1&nbsp;&nbsp;</span>Data curation</a></span></li><li><span><a href="#Decode" data-toc-modified-id="Decode-2.6.4.2"><span class="toc-item-num">2.6.4.2&nbsp;&nbsp;</span>Decode</a></span></li><li><span><a href="#Curate-dementia-related-variables" data-toc-modified-id="Curate-dementia-related-variables-2.6.4.3"><span class="toc-item-num">2.6.4.3&nbsp;&nbsp;</span>Curate dementia-related variables</a></span></li><li><span><a href="#Normalising-CRP-and-WMH" data-toc-modified-id="Normalising-CRP-and-WMH-2.6.4.4"><span class="toc-item-num">2.6.4.4&nbsp;&nbsp;</span>Normalising CRP and WMH</a></span></li><li><span><a href="#Save" data-toc-modified-id="Save-2.6.4.5"><span class="toc-item-num">2.6.4.5&nbsp;&nbsp;</span>Save</a></span></li><li><span><a href="#Combine-with-previous-dataset,-adding-blood-types" data-toc-modified-id="Combine-with-previous-dataset,-adding-blood-types-2.6.4.6"><span class="toc-item-num">2.6.4.6&nbsp;&nbsp;</span>Combine with previous dataset, adding blood types</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#Analysis:-COVID-19-cases" data-toc-modified-id="Analysis:-COVID-19-cases-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Analysis: COVID-19 cases</a></span><ul class="toc-item"><li><span><a href="#Descriptive-statistics---data-in-models-only" data-toc-modified-id="Descriptive-statistics---data-in-models-only-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Descriptive statistics - data in models only</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Distributions-of-each-variable" data-toc-modified-id="Distributions-of-each-variable-3.1.0.1"><span class="toc-item-num">3.1.0.1&nbsp;&nbsp;</span>Distributions of each variable</a></span></li></ul></li></ul></li><li><span><a href="#Initial-model:-binomial-linear-model-with-all-variables,-for-infection" data-toc-modified-id="Initial-model:-binomial-linear-model-with-all-variables,-for-infection-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Initial model: binomial linear model with all variables, for infection</a></span><ul class="toc-item"><li><span><a href="#Model-analysis" data-toc-modified-id="Model-analysis-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Model analysis</a></span></li><li><span><a href="#Simplification-of-the-full-model" data-toc-modified-id="Simplification-of-the-full-model-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Simplification of the full model</a></span><ul class="toc-item"><li><span><a href="#Comparison-of-the-simplified-model-versus-the-full-model" data-toc-modified-id="Comparison-of-the-simplified-model-versus-the-full-model-3.2.2.1"><span class="toc-item-num">3.2.2.1&nbsp;&nbsp;</span>Comparison of the simplified model versus the full model</a></span></li><li><span><a href="#Update-the-model-with-the-full-dataset." data-toc-modified-id="Update-the-model-with-the-full-dataset.-3.2.2.2"><span class="toc-item-num">3.2.2.2&nbsp;&nbsp;</span>Update the model with the full dataset.</a></span></li><li><span><a href="#Create-a-model-with-only-dementia---all-diagnosis" data-toc-modified-id="Create-a-model-with-only-dementia---all-diagnosis-3.2.2.3"><span class="toc-item-num">3.2.2.3&nbsp;&nbsp;</span>Create a model with only dementia - all diagnosis</a></span></li><li><span><a href="#Show-the-importance-of-dementia-related-factors" data-toc-modified-id="Show-the-importance-of-dementia-related-factors-3.2.2.4"><span class="toc-item-num">3.2.2.4&nbsp;&nbsp;</span>Show the importance of dementia-related factors</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#Curation-and-analysis-of-COVID-19-death-data" data-toc-modified-id="Curation-and-analysis-of-COVID-19-death-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Curation and analysis of COVID-19 death data</a></span><ul class="toc-item"><li><span><a href="#Data-curation-and-merging" data-toc-modified-id="Data-curation-and-merging-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Data curation and merging</a></span><ul class="toc-item"><li><span><a href="#Calculate-the-percentage-of-deaths-related-to-COVID-19-&amp;-odds-ratio" data-toc-modified-id="Calculate-the-percentage-of-deaths-related-to-COVID-19-&amp;-odds-ratio-4.1.1"><span class="toc-item-num">4.1.1&nbsp;&nbsp;</span>Calculate the percentage of deaths related to COVID-19 &amp; odds ratio</a></span></li><li><span><a href="#Mortality-odds-while-acccounting-for-other-factors-including-COVID-19" data-toc-modified-id="Mortality-odds-while-acccounting-for-other-factors-including-COVID-19-4.1.2"><span class="toc-item-num">4.1.2&nbsp;&nbsp;</span>Mortality odds while acccounting for other factors including COVID-19</a></span><ul class="toc-item"><li><span><a href="#Scale-all-numeric-data-and-curate-ethnicity-value" data-toc-modified-id="Scale-all-numeric-data-and-curate-ethnicity-value-4.1.2.1"><span class="toc-item-num">4.1.2.1&nbsp;&nbsp;</span>Scale all numeric data and curate ethnicity value</a></span></li><li><span><a href="#General-model-on-death,-all-causes" data-toc-modified-id="General-model-on-death,-all-causes-4.1.2.2"><span class="toc-item-num">4.1.2.2&nbsp;&nbsp;</span>General model on death, all causes</a></span></li></ul></li><li><span><a href="#Main-contributors-of-COVID-19-related-death" data-toc-modified-id="Main-contributors-of-COVID-19-related-death-4.1.3"><span class="toc-item-num">4.1.3&nbsp;&nbsp;</span>Main contributors of COVID-19-related death</a></span><ul class="toc-item"><li><span><a href="#Dementia-only-model---COVID-death" data-toc-modified-id="Dementia-only-model---COVID-death-4.1.3.1"><span class="toc-item-num">4.1.3.1&nbsp;&nbsp;</span>Dementia-only model - COVID death</a></span></li><li><span><a href="#Dementia-subtypes" data-toc-modified-id="Dementia-subtypes-4.1.3.2"><span class="toc-item-num">4.1.3.2&nbsp;&nbsp;</span>Dementia subtypes</a></span></li><li><span><a href="#Simplification-of-the-COVID-death-model" data-toc-modified-id="Simplification-of-the-COVID-death-model-4.1.3.3"><span class="toc-item-num">4.1.3.3&nbsp;&nbsp;</span>Simplification of the COVID death model</a></span></li><li><span><a href="#Simplified-COVID-death-model" data-toc-modified-id="Simplified-COVID-death-model-4.1.3.4"><span class="toc-item-num">4.1.3.4&nbsp;&nbsp;</span>Simplified COVID death model</a></span></li><li><span><a href="#AD-model" data-toc-modified-id="AD-model-4.1.3.5"><span class="toc-item-num">4.1.3.5&nbsp;&nbsp;</span>AD model</a></span></li><li><span><a href="#White-matter-hypersintensity-model" data-toc-modified-id="White-matter-hypersintensity-model-4.1.3.6"><span class="toc-item-num">4.1.3.6&nbsp;&nbsp;</span>White matter hypersintensity model</a></span></li></ul></li><li><span><a href="#AD-subset-model" data-toc-modified-id="AD-subset-model-4.1.4"><span class="toc-item-num">4.1.4&nbsp;&nbsp;</span>AD-subset model</a></span></li></ul></li><li><span><a href="#Add-drug-intake-data" data-toc-modified-id="Add-drug-intake-data-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Add drug-intake data</a></span><ul class="toc-item"><li><span><a href="#Subset-the-data-to-decrease-processing-time" data-toc-modified-id="Subset-the-data-to-decrease-processing-time-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>Subset the data to decrease processing time</a></span></li><li><span><a href="#Create-separate-columns-for-each-of-the-8-drugs-investigated" data-toc-modified-id="Create-separate-columns-for-each-of-the-8-drugs-investigated-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>Create separate columns for each of the 8 drugs investigated</a></span></li><li><span><a href="#Check-how-many-people-take-each-drug-in-the-drug-dataset" data-toc-modified-id="Check-how-many-people-take-each-drug-in-the-drug-dataset-4.2.3"><span class="toc-item-num">4.2.3&nbsp;&nbsp;</span>Check how many people take each drug in the drug dataset</a></span></li><li><span><a href="#Merge-and-save" data-toc-modified-id="Merge-and-save-4.2.4"><span class="toc-item-num">4.2.4&nbsp;&nbsp;</span>Merge and save</a></span></li></ul></li><li><span><a href="#Relationship-between-9-drugs-and-COVID-19-related-death" data-toc-modified-id="Relationship-between-9-drugs-and-COVID-19-related-death-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Relationship between 9 drugs and COVID-19 related death</a></span><ul class="toc-item"><li><span><a href="#Descriptive-stats-for-the-drugs" data-toc-modified-id="Descriptive-stats-for-the-drugs-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>Descriptive stats for the drugs</a></span></li><li><span><a href="#Subset-descriptive-statistics-for-dementia" data-toc-modified-id="Subset-descriptive-statistics-for-dementia-4.3.2"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>Subset descriptive statistics for dementia</a></span></li><li><span><a href="#Susbet-amantidine" data-toc-modified-id="Susbet-amantidine-4.3.3"><span class="toc-item-num">4.3.3&nbsp;&nbsp;</span>Susbet amantidine</a></span></li></ul></li></ul></li><li><span><a href="#AD-/-PD-drug-analysis" data-toc-modified-id="AD-/-PD-drug-analysis-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>AD / PD drug analysis</a></span><ul class="toc-item"><li><span><a href="#Data-curation" data-toc-modified-id="Data-curation-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Data curation</a></span><ul class="toc-item"><li><span><a href="#Subset-AD-&amp;-PD-eids" data-toc-modified-id="Subset-AD-&amp;-PD-eids-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Subset AD &amp; PD eids</a></span></li><li><span><a href="#Overview-of-drugs-taken-by-these-participants" data-toc-modified-id="Overview-of-drugs-taken-by-these-participants-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Overview of drugs taken by these participants</a></span></li><li><span><a href="#Curate-drugs-with-sufficient-data" data-toc-modified-id="Curate-drugs-with-sufficient-data-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span>Curate drugs with sufficient data</a></span></li></ul></li><li><span><a href="#Analysis---death" data-toc-modified-id="Analysis---death-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Analysis - death</a></span><ul class="toc-item"><li><span><a href="#Data-visualisation" data-toc-modified-id="Data-visualisation-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Data visualisation</a></span></li><li><span><a href="#AD-drug-model" data-toc-modified-id="AD-drug-model-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>AD drug model</a></span><ul class="toc-item"><li><span><a href="#Complex-model" data-toc-modified-id="Complex-model-5.2.2.1"><span class="toc-item-num">5.2.2.1&nbsp;&nbsp;</span>Complex model</a></span></li><li><span><a href="#Models-with-only-age,-sex-and-whr-as-covariates" data-toc-modified-id="Models-with-only-age,-sex-and-whr-as-covariates-5.2.2.2"><span class="toc-item-num">5.2.2.2&nbsp;&nbsp;</span>Models with only age, sex and whr as covariates</a></span></li><li><span><a href="#raw-odds-of-drugs-in-the-AD-models" data-toc-modified-id="raw-odds-of-drugs-in-the-AD-models-5.2.2.3"><span class="toc-item-num">5.2.2.3&nbsp;&nbsp;</span>raw odds of drugs in the AD models</a></span></li></ul></li><li><span><a href="#PD-drug-models" data-toc-modified-id="PD-drug-models-5.2.3"><span class="toc-item-num">5.2.3&nbsp;&nbsp;</span>PD drug models</a></span><ul class="toc-item"><li><span><a href="#Complex-model" data-toc-modified-id="Complex-model-5.2.3.1"><span class="toc-item-num">5.2.3.1&nbsp;&nbsp;</span>Complex model</a></span></li><li><span><a href="#Simplified-model" data-toc-modified-id="Simplified-model-5.2.3.2"><span class="toc-item-num">5.2.3.2&nbsp;&nbsp;</span>Simplified model</a></span></li><li><span><a href="#Model-with-only-drug---raw-odds" data-toc-modified-id="Model-with-only-drug---raw-odds-5.2.3.3"><span class="toc-item-num">5.2.3.3&nbsp;&nbsp;</span>Model with only drug - raw odds</a></span></li></ul></li></ul></li><li><span><a href="#Analysis---infection" data-toc-modified-id="Analysis---infection-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Analysis - infection</a></span><ul class="toc-item"><li><span><a href="#AD-infection-models" data-toc-modified-id="AD-infection-models-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>AD infection models</a></span><ul class="toc-item"><li><span><a href="#Visualisation" data-toc-modified-id="Visualisation-5.3.1.1"><span class="toc-item-num">5.3.1.1&nbsp;&nbsp;</span>Visualisation</a></span></li><li><span><a href="#Complex-model" data-toc-modified-id="Complex-model-5.3.1.2"><span class="toc-item-num">5.3.1.2&nbsp;&nbsp;</span>Complex model</a></span></li><li><span><a href="#Simplified-model" data-toc-modified-id="Simplified-model-5.3.1.3"><span class="toc-item-num">5.3.1.3&nbsp;&nbsp;</span>Simplified model</a></span></li></ul></li><li><span><a href="#PD-infection-models" data-toc-modified-id="PD-infection-models-5.3.2"><span class="toc-item-num">5.3.2&nbsp;&nbsp;</span>PD infection models</a></span><ul class="toc-item"><li><span><a href="#Complex-models" data-toc-modified-id="Complex-models-5.3.2.1"><span class="toc-item-num">5.3.2.1&nbsp;&nbsp;</span>Complex models</a></span></li><li><span><a href="#Simplified-model" data-toc-modified-id="Simplified-model-5.3.2.2"><span class="toc-item-num">5.3.2.2&nbsp;&nbsp;</span>Simplified model</a></span></li></ul></li></ul></li></ul></li></ul></div>

# Data Curation

## Set R environment & Load python libraries 


```python
import pandas as pd
import tensorflow as tf
#load R 
%load_ext rpy2.ipython
```

    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/robjects/pandas2ri.py:17: FutureWarning: pandas.core.index is deprecated and will be removed in a future version.  The public classes are available in the top-level namespace.
      from pandas.core.index import Index as PandasIndex


## Obtain the UK Biobank phenotype data subset, with the variables of interest.

### Create a selector function



```python
def UKB_select_rows(input_dt,col_list,output_dt):
    #input_dt = full UKB dataset
    #col_list = metadata 
    #output_dt = name of output data

    input_name = str(''.join(input_dt))
    output_name = str(''.join(output_dt))
    col_list_name = str(''.join(col_list))

    #create the list of columns to select 
    col_df = pd.read_csv(col_list_name, sep = ',', usecols = ["FieldID","Instances",'Array'])
    
    #this is the array of columns to select, dont forget to include eid...
    col_list_selected = ['eid']

    #here I make a loop to collect all instances + arrays of a field
    for index, row in col_df.iterrows():
        # range automatically goes from 0 to n-1
        instance_list = list(range(0,row['Instances'].astype(int)))
        list_single_field = []
        for instance_item in instance_list:
            field_instance = [row['FieldID'].astype(str) + "-" + str(instance_item)]
            list_single_field = list_single_field + field_instance
            array_list = list(range(0,row['Array'].astype(int)))
            for fi_item in list_single_field:
                for array_item in array_list:
                    field_instance_array = [str(fi_item) + "." + str(array_item)]
                    col_list_selected = col_list_selected + field_instance_array
                
    #print(len(col_list_selected))
    #col_df['UID'] = col_df["FieldID"].astype(str) + "-" + col_df["Instances"].astype(str) + "." + col_df['Array'].astype(str)
    #selected_UID_list = col_df['UID'].tolist()
    
    # read the large csv file with specified chunksize 
    #I tried with many chunksizes... 
    df_chunk = pd.read_csv(input_name, sep = ',', chunksize=10000, dtype={'eid': int},encoding= 'unicode_escape', usecols = col_list_selected)

    # Each chunk is in df format -> save them 
    # only write header once...
    write_header = True
    for chunk in df_chunk:
        print("10K done, still running...")
        chunk.to_csv(output_name, mode='a', index=False, header=write_header)
        write_header = False
```


```python
UKB_select_rows(input_dt="/Users/yizhouyu/camDrive/UKB_dt_18-4-2020/ukb41646.csv", 
                col_list = "/Users/yizhouyu/camDrive/Miguel_shared_Yizhou/COVID_project_yizhou/AirPoll_UKB_6000/ukb_covid19/data/UKB_metadt_AIRcovid2.csv", 
                output_dt = "/Users/yizhouyu/camDrive/Miguel_shared_Yizhou/COVID_project_yizhou/AirPoll_UKB_6000/ukb_covid19/data_output/UKB_subset_AP_analysis.csv")

```

    /Users/yizhouyu/py_37_env/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3254: DtypeWarning: Columns (16450,16460,16462) have mixed types.Specify dtype option on import or set low_memory=False.
      if (await self.run_code(code, result,  async_=asy)):


    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...


### Structure of the curated data

The data has still remains quite large


```python
# pd.read_csv("/Users/yizhouyu/camDrive/Miguel_shared_Yizhou/COVID_project_yizhou/AirPoll_UKB_6000/ukb_covid19/data_output/UKB_subset_AP_analysis.csv").head()
```

## Subset the participants with COVID-19 results (as of August 2020)

### Load UK Biobank COVID-19 data



```r
%%R 
raw_ukb_covid = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/covid_refresh_16_8_2020/covid19_result_17_8_2020.txt", sep = "\t")
nrow(raw_ukb_covid)
```


    [1] 20554



#### Get the date range of the test results 


```r
%%R
raw_ukb_covid$specdate = as.Date(raw_ukb_covid$specdate,tryFormats = "%d/%m/%Y")
max(raw_ukb_covid$specdate)
```


    [1] "2020-07-26"



For the the positive cases, I would like to know when they are first tested positive, i.e. the earliest date only. 


```r
%%R 
raw_ukb_covid.pos = subset(raw_ukb_covid, result == 1)
raw_ukb_covid.pos_cur = aggregate(raw_ukb_covid.pos, list(raw_ukb_covid.pos$eid), FUN=head, 1)
```

For the negative test results, I should keep the last date because that is the most up-to-date test. 


```r
%%R 
raw_ukb_covid.neg = subset(raw_ukb_covid, result == 0)
raw_ukb_covid.neg_cur = aggregate(raw_ukb_covid.neg, list(raw_ukb_covid.neg$eid), FUN=tail, 1)
```


```r
%%R
ukb_covid_bind = rbind(raw_ukb_covid.pos_cur, raw_ukb_covid.neg_cur)
```


```r
%%R
print(paste(length((ukb_covid_bind$eid)),length(unique(ukb_covid_bind$eid)),sep = " "))
```


    [1] "13940 13502"



There are still ~400 eids that are duplicates. I will sum them to keep the positive tests only, with the latest date.


```r
%%R
ukb_covid_bind = ukb_covid_bind[c("eid","specdate","result")]
ukb_covid_bind$specdate = as.Date(ukb_covid_bind$specdate,tryFormats = "%d/%m/%Y")
ukb_covid_bind_test = aggregate(ukb_covid_bind, list(ukb_covid_bind$eid), FUN=max)[c("eid","specdate","result")]
nrow(ukb_covid_bind_test)
```


    [1] 13502



### Save curated data

The curated data now is: if a participant is positive, then save this result & date, if not, save the latest (most up-to-date) negative test result.


```r
%%R
write.csv(ukb_covid_bind_test,"data_output/curated_ukb_covid_test_results.csv", row.names=FALSE)
```

## Mergeing the 2 parts of this dataset

### Here, I will get the most up-to-date result

### Subset the data with only the COVID-19-tested participants

#### Select rows by chunk


```python
def UKB_select_rows(input_dt,row_list,output_dt):
    input_name = str(''.join(input_dt))
    row_list_name = str(''.join(row_list))
    output_name = str(''.join(output_dt))

    #get the list of IDs
    ID_list_df = pd.read_csv(row_list_name, sep = ',', usecols = ['eid'], dtype={'eid': int})
    selected_ID_list = ID_list_df['eid'].tolist()
    #print(selected_ID_list)

    df_chunk = pd.read_csv(input_name, sep = ',', chunksize=10000, 
                           encoding= 'unicode_escape')

    #make a chunks list 
    chunk_list = []
    # Each chunk is in df format
    for chunk in df_chunk:  
        # perform data filtering 
        chunk_filter = chunk[chunk['eid'].isin(selected_ID_list)]
        print("10K done, still running...")
        #get the rows that are in the ID list
        #print(chunk_filter)
        # Once the data filtering is done, append the chunk to list
        chunk_list.append(chunk_filter)
        
    # concat the list into dataframe 
    #header_chunk.append(chunk_list)
    df_concat = pd.concat(chunk_list)
    #print(df_concat)
    df_concat.to_csv(output_name)
```

#### Select rows by chunk


```python
UKB_select_rows(input_dt="data_output/UKB_subset_AP_analysis.csv",
                row_list="data_output/curated_ukb_covid_test_results.csv",
               output_dt="data_output/full_ukb_covidTested_results.csv")
```

    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3254: DtypeWarning: Columns (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,276,278,280) have mixed types.Specify dtype option on import or set low_memory=False.
      if (await self.run_code(code, result,  async_=asy)):


    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...


    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3254: DtypeWarning: Columns (275) have mixed types.Specify dtype option on import or set low_memory=False.
      if (await self.run_code(code, result,  async_=asy)):


    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...
    10K done, still running...


### Get the most up-to-date data from these participants



```r
%%R
# load non-covid UKB phenotype data
ukb_pheno_dt = read.csv("data_output/full_ukb_covidTested_results.csv")[-1]
#-1: delete first column of index from the previous function

#aggregate the columns by selecting the last results, for each 
library(reshape)
ukb_pheno_dt_t = melt(ukb_pheno_dt, id='eid')
# head(ukb_pheno_dt_t)

#order by variable to make sure I am will be replacing the latest value 
ukb_pheno_dt_t <- ukb_pheno_dt_t[order(ukb_pheno_dt_t$variable),] 

#delete everything after period 
ukb_pheno_dt_t$variable = gsub("\\..*","",ukb_pheno_dt_t$variable)

#aggregate by last 
ukb_covid_t_na = na.omit(ukb_pheno_dt_t)
ukb_covid_t_na = aggregate(ukb_covid_t_na, by=list(ukb_covid_t_na$eid,ukb_covid_t_na$variable), FUN=tail, n = 1)

# Curate
ukb_covid_t_na = ukb_covid_t_na[c("eid", "variable", "value")]
ukb_covid_t_na$variable = as.factor(ukb_covid_t_na$variable)

# put it back straight 
ukb_covid_cur = cast(ukb_covid_t_na, eid~variable)

```

### Decode with the UK Biobank values

#### Prepare encodings


```r
%%R
#get levels and labels 
lvl.0493 <- c(-141,-131,-121,0)
lbl.0493 <- c("Often","Sometimes","Do not know","Rarely/never")

lvl.0007 <- c(0,1)
lbl.0007 <- c("FALSE","TRUE")

lvl.100349 <- c(-3,-1,0,1)
lbl.100349 <- c("Prefer not to answer","Do not know","FALSE","TRUE")

lvl.0090 <- c(-3,0,1,2)
lbl.0090 <- c("Prefer not to answer","Never","Previous","Current")

lvl.0009 <- c(0,1)
lbl.0009 <- c("Female","Male")

lvl.0272 <- c(1)
lbl.0272 <- c("Date is unknown")

lvl.0300 <- c(0,1,2)
lbl.0300 <- c("Self-reported only","Hospital admission","Death only")

lvl.100290 <- c(-10,-3,-1)
lbl.100290 <- c("Less than a year","Prefer not to answer","Do not know")

lvl.100291 <- c(-3,-1)
lbl.100291 <- c("Prefer not to answer","Do not know")

lvl.100294 <- c(-3,-1,1,2,3,4,5)
lbl.100294 <- c("Prefer not to answer","Do not know",
                "Less than 18,000","18,000 to 30,999","31,000 to 51,999",
                "52,000 to 100,000","Greater than 100,000")

lvl.100298 <- c(-10,-3,-1)
lbl.100298 <- c("Less than once a week","Prefer not to answer","Do not know")

lvl.1001 <- c(-3,-1,1001,1002,1003,2001,2002,2003,2004,3001,3002,3003,3004,4001,4002,4003,1,2,3,4,5,6)
lbl.1001 <- c("Prefer not to answer","Do not know",
              "British","Irish","Any other white background","White and Black Caribbean",
              "White and Black African","White and Asian","Any other mixed background","Indian",
              "Pakistani","Bangladeshi","Any other Asian background","Caribbean","African","Any other Black background",
              "White","Mixed","Asian or Asian British","Black or Black British","Chinese","Other ethnic group")

### eid replacement, saved here ----
yy_replace <- function(string, patterns, replacements) {
  for (i in seq_along(patterns))
    string <- gsub(patterns[i], replacements[i], string, perl=TRUE)
  string
}
#label = replacement, lvl = to replace

```

#### Decode UKB phenotype data


```r
%%R
ukb_covid_decode = ukb_covid_cur
```


```r
%%R
ukb_covid_decode$X22609 <- yy_replace(ukb_covid_decode$X22609, lvl.0493, lbl.0493)
ukb_covid_decode$X22610 <- yy_replace(ukb_covid_decode$X22610, lvl.0493, lbl.0493)
ukb_covid_decode$X22611 <- yy_replace(ukb_covid_decode$X22611, lvl.0493, lbl.0493)
ukb_covid_decode$X22615 <- yy_replace(ukb_covid_decode$X22615, lvl.0493, lbl.0493)
ukb_covid_decode$X22616 <- yy_replace(ukb_covid_decode$X22616, lvl.0007, lbl.0007)
ukb_covid_decode$X2316 <- yy_replace(ukb_covid_decode$X2316, lvl.100349, lbl.100349)
ukb_covid_decode$X31 <- yy_replace(ukb_covid_decode$X31, lvl.0009, lbl.0009)
ukb_covid_decode$X22130 <- yy_replace(ukb_covid_decode$X22130, lvl.0007, lbl.0007)
ukb_covid_decode$X2443 <- yy_replace(ukb_covid_decode$X2443, lvl.100349, lbl.100349)
ukb_covid_decode$X20116 <- yy_replace(ukb_covid_decode$X20116, lvl.0090, lbl.0090)

ukb_covid_decode$X42030 <- yy_replace(ukb_covid_decode$X42030, lvl.0272, lbl.0272)

ukb_covid_decode$X42031 <- yy_replace(ukb_covid_decode$X42031, lvl.0300, lbl.0300)

ukb_covid_decode$X42032 <- yy_replace(ukb_covid_decode$X42032, lvl.0272, lbl.0272)

ukb_covid_decode$X42033 <- yy_replace(ukb_covid_decode$X42033, lvl.0300, lbl.0300)

#ukb_covid_cur$X100291 <- yy_replace(ukb_covid_cur$X100291, lvl.100290, lbl.100290)

ukb_covid_decode$X709 <- yy_replace(ukb_covid_decode$X709, lvl.100291, lbl.100291)

ukb_covid_decode$X777 <- yy_replace(ukb_covid_decode$X777, lvl.100298, lbl.100298)

#household income

ukb_covid_decode$X738 <- ordered(ukb_covid_decode$X738, levels = lvl.100294, 
                                 labels = lbl.100294)

#ethnicity
ukb_covid_decode$X21000 <- ordered(ukb_covid_decode$X21000, levels = lvl.1001, 
                                  labels = lbl.1001)

ukb_covid_decode$X42021 <- yy_replace(ukb_covid_decode$X42021, lvl.0090, lbl.0090)

ukb_covid_decode$X42020 <- yy_replace(ukb_covid_decode$X42020, lvl.0272, lbl.0272)
```

#### Assign custom column names


```r
%%R
#get the ID that I designed
UID_names = read.csv("data/UKB_metadt_AIRcovid2.csv")[c("FieldID","my_colname")]
UID_names$FieldID = paste("X",UID_names$FieldID, sep = "")
UID_names$my_colname = lapply(UID_names$my_colname,toString)
UID_names[nrow(UID_names) + 1,] = c("eid","eid")

# sort it according to the dataset

UID_names_match = colnames(ukb_covid_cur)

UID_sort_df = data.frame(UID_names_match)
colnames(UID_sort_df) <- "FieldID"

#merge to sort
UID_sort_df = merge(UID_sort_df, UID_names, by = "FieldID")
# this is now sorted according to the correct df

#replace column names
ukb_covid_pheno_df = ukb_covid_decode

colnames(ukb_covid_pheno_df) <- UID_sort_df$my_colname
head(ukb_covid_pheno_df)
```


          eid n_cancers    townsend x_coord y_coord  smoking     fev ethnicity
    1 1016273         0    -5.87237  441000  214000 Previous    <NA>   British
    2 1016298         0     2.91871  420000  562000 Previous    <NA>   British
    3 1016708         0     3.43245  532000  168000    Never    <NA> Caribbean
    4 1017007         0    -2.30303  396000   4e+05 Previous    <NA>   British
    5 1017365         0 -0.00099431  408000  282000 Previous    <NA>   British
    6 1017713         0    -3.92656  329000  407000    Never 3.11074   British
       copd     WP_dusty WP_chemicals       WP_cig    WP_diesel breathing whistling
    1  <NA>         <NA>         <NA>         <NA>         <NA>      <NA>     FALSE
    2 FALSE Rarely/never Rarely/never Rarely/never Rarely/never     FALSE     FALSE
    3  <NA>         <NA>         <NA>         <NA>         <NA>      <NA>     FALSE
    4  <NA>         <NA>         <NA>         <NA>         <NA>      <NA>     FALSE
    5 FALSE Rarely/never Rarely/never Rarely/never Rarely/never     FALSE     FALSE
    6  <NA>         <NA>         <NA>         <NA>         <NA>      <NA>     FALSE
      diabetes    sex birthYear diaBP sysBP ad_date   ad pd_date_1 pd_1 pd_date_2
    1    FALSE   Male      1950    76   114         <NA>           <NA>          
    2    FALSE Female      1948    81   143         <NA>           <NA>          
    3    FALSE Female      1967    61    95         <NA>           <NA>          
    4    FALSE   Male      1939    96   163         <NA>           <NA>          
    5    FALSE Female      1947    89   138         <NA>           <NA>          
    6    FALSE   Male      1945    85   156         <NA>           <NA>          
      pd_2 waist hip height yearAtlocation    peopleInHousehold
    1 <NA>    86 101    171             10                    2
    2 <NA>   100 110    154              6                    2
    3 <NA>    93 104    175              5                    1
    4 <NA>   103 107    175             32                    2
    5 <NA>    93 101    152             33                    1
    6 <NA>   120 117    172             -3 Prefer not to answer
      AverageHouseholdIncome travelToWork
    1   Prefer not to answer            4
    2       18,000 to 30,999            4
    3            Do not know         <NA>
    4       18,000 to 30,999         <NA>
    5       18,000 to 30,999         <NA>
    6   Prefer not to answer         <NA>



### Merge the UK Biobank phenotype data with the UK Biobank COVID-19 data


```r
%%R 
ukb_covid_pheno_merged = merge(ukb_covid_pheno_df, ukb_covid_bind_test, by = "eid")
write.csv(ukb_covid_pheno_merged,"data_output/ukb_covid19_pheno_merged.csv", row.names=FALSE)
```

#### Curate the PD/AD diagnosis data


```r
%%R 
ukb_covid_pheno_merged = read.csv("data_output/ukb_covid19_pheno_merged.csv")
# PD diagnosis
library(dplyr)
ukb_covid_pheno_PDcur = ukb_covid_pheno_merged %>%
  mutate(PD_diag = c("FALSE", "TRUE")[(!is.na(pd_1) | 
                                     !is.na(pd_2) 
                                     )+1] )
# Ad diagnosis
ukb_covid_pheno_PDADcur = ukb_covid_pheno_PDcur %>%
  mutate(AD_diag = c("FALSE", "TRUE")[(!is.na(ad)
                                     )+1] )
```

    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: 
    Attaching package: ‘dplyr’
    
    
      warnings.warn(x, RRuntimeWarning)
    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: The following object is masked from ‘package:reshape’:
    
        rename
    
    
      warnings.warn(x, RRuntimeWarning)
    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    
      warnings.warn(x, RRuntimeWarning)
    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    
    
      warnings.warn(x, RRuntimeWarning)


#### Hypertension / high blood pressure calculations


```r
%%R
ukb_covid_pheno_PDADcur = ukb_covid_pheno_PDADcur %>%
  mutate(highBP = c("FALSE", "TRUE")[(diaBP >= 90 | sysBP > 140
                                     )+1] )
```

#### Calculate Age: 2020 - birth year


```r
%%R 
ukb_covid_pheno_PDADcur$age = 2020 - ukb_covid_pheno_PDADcur$birthYear
```

#### Drop the unused columns


```r
%%R 
vars_drop = c("specdate","X","yearAtlocation",
              "ad_date","ad","pd_date_1",
              "pd_1","pd_date_2","pd_2","diaBP","sysBP")
ukb_covid_pheno_subset = ukb_covid_pheno_PDADcur[ , !(names(ukb_covid_pheno_PDADcur) %in% vars_drop)]
```


```r
%%R 
# Ad diagnosis
head(ukb_covid_pheno_subset)
```


          eid n_cancers    townsend x_coord y_coord  smoking     fev ethnicity
    1 1016273         0 -5.87237000  441000  214000 Previous      NA   British
    2 1016298         0  2.91871000  420000  562000 Previous      NA   British
    3 1016708         0  3.43245000  532000  168000    Never      NA Caribbean
    4 1017007         0 -2.30303000  396000  400000 Previous      NA   British
    5 1017365         0 -0.00099431  408000  282000 Previous      NA   British
    6 1017713         0 -3.92656000  329000  407000    Never 3.11074   British
       copd     WP_dusty WP_chemicals       WP_cig    WP_diesel breathing whistling
    1    NA         <NA>         <NA>         <NA>         <NA>        NA     FALSE
    2 FALSE Rarely/never Rarely/never Rarely/never Rarely/never     FALSE     FALSE
    3    NA         <NA>         <NA>         <NA>         <NA>        NA     FALSE
    4    NA         <NA>         <NA>         <NA>         <NA>        NA     FALSE
    5 FALSE Rarely/never Rarely/never Rarely/never Rarely/never     FALSE     FALSE
    6    NA         <NA>         <NA>         <NA>         <NA>        NA     FALSE
      diabetes    sex birthYear waist hip height    peopleInHousehold
    1    FALSE   Male      1950    86 101    171                    2
    2    FALSE Female      1948   100 110    154                    2
    3    FALSE Female      1967    93 104    175                    1
    4    FALSE   Male      1939   103 107    175                    2
    5    FALSE Female      1947    93 101    152                    1
    6    FALSE   Male      1945   120 117    172 Prefer not to answer
      AverageHouseholdIncome travelToWork result PD_diag AD_diag highBP age
    1   Prefer not to answer            4      0   FALSE   FALSE  FALSE  70
    2       18,000 to 30,999            4      0   FALSE   FALSE   TRUE  72
    3            Do not know         <NA>      0   FALSE   FALSE  FALSE  53
    4       18,000 to 30,999         <NA>      0   FALSE   FALSE   TRUE  81
    5       18,000 to 30,999         <NA>      0   FALSE   FALSE  FALSE  73
    6   Prefer not to answer         <NA>      0   FALSE   FALSE   TRUE  75




```r
%%R
write.csv(ukb_covid_pheno_subset,"data_output/merged_ukb_covid_test_results.csv", row.names=FALSE)
```

### Create Descriptive table - obsolete


```python
# %%R
# library(arsenal)
# ukb_covid_pheno_subset = read.csv("data_output/merged_ukb_covid_test_results.csv")
# ukb_descript_stats <- tableby(result ~ ., data = ukb_covid_pheno_subset[,2:ncol(ukb_covid_pheno_subset)])
# write2pdf(ukb_descript_stats, "ukb_descript_stats.pdf",
#   keep.md = TRUE,
#   quiet = TRUE, # passed to rmarkdown::render
#   title = "Descriptive statistics of the UK Biobank data") # passed to summary.tableby
# write2word(ukb_descript_stats, "ukb_descript_stats.doc",
#   keep.md = TRUE,
#   quiet = TRUE, # passed to rmarkdown::render
#   title = "Descriptive statistics of the UK Biobank data") # passed to summary.tableby
```

## Add the exposure to air pollution

The raw air pollution files were obtained from DEFRA, UK: https://uk-air.defra.gov.uk/data/pcm-data.
<br> The raw X Y location values were converted to long-lat using an R code from Travaglio *et. al.*, 2020, available here: https://github.com/M1gus/AirPollutionCOVID19/. 
<br> The air pollution values are from 2018.

### Convert UKB X & Y to long lat


```r
%%R
ukb_covid_pheno_subset = read.csv("data_output/merged_ukb_covid_test_results.csv")
ukgrid <- "+init=epsg:27700"
latlong <- "+init=epsg:4326"

ukb_coords = na.omit(ukb_covid_pheno_subset[c("eid","x_coord","y_coord")])
ukb_coords_NE <- data.frame(cbind(Easting = as.numeric(as.character(ukb_coords$x_coord)),
                     Northing = as.numeric(as.character(ukb_coords$y_coord))))
### Create the SpatialPointsDataFrame
library(sp)
ukb_covid_SP <- SpatialPointsDataFrame(ukb_coords_NE,
                                  data = ukb_coords,
                                  proj4string = CRS("+init=epsg:27700"))
```


```r
%%R
### Convert
library(rgdal)
ukb_covid_ll <- spTransform(ukb_covid_SP, CRS(latlong))

ukb_covid_ll_df = data.frame('ukb_lon' = coordinates(ukb_covid_ll)[, 1], 
                             'ukb_lat' = coordinates(ukb_covid_ll)[, 2], 
                             'eid' = ukb_covid_ll$eid)
write.csv(ukb_covid_ll_df, ("data_output/ukb_covid_lonlat_df.csv"))
# plot UKB locations
```

    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: rgdal: version: 1.4-4, (SVN revision 833)
     Geospatial Data Abstraction Library extensions to R successfully loaded
     Loaded GDAL runtime: GDAL 2.4.2, released 2019/06/28
     Path to GDAL shared files: /Users/yizhouyu/miniconda3/share/gdal
     GDAL binary built with GEOS: TRUE 
     Loaded PROJ.4 runtime: Rel. 6.1.0, May 15th, 2019, [PJ_VERSION: 610]
     Path to PROJ.4 shared files: /Users/yizhouyu/miniconda3/share/proj
     Linking to sp version: 1.3-1 
    
      warnings.warn(x, RRuntimeWarning)


### Graph the location of the UK Biobank participants


```python
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import pandas as pd

ukb_covid_ll_df = pd.read_csv("data_output/ukb_covid_lonlat_df.csv")

geometry = [Point(xy) for xy in zip(ukb_covid_ll_df['ukb_lon'], ukb_covid_ll_df['ukb_lat'])]
gdf = GeoDataFrame(ukb_covid_ll_df, geometry=geometry)   

#this is a simple map that goes with geopandas
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
UK_map = cities[cities["name"] == 'United Kingdom']
# gdf.plot(ax=UK_map.plot(figsize=(10, 6)), marker='o', color='red', markersize=15)
```


```python
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
UK_map = cities[cities["name"] == 'United Kingdom']

ukb_participants_plt = gdf.plot(ax=UK_map.plot(figsize=(10, 6)), marker='.', color='red', markersize=15)
ukb_participants_fig = ukb_participants_plt.get_figure()
ukb_participants_fig.savefig('fig_out/ukb_participants_plt.pdf')
```


![png](output_57_0.png)


### Match the location of air pollution levels to location of UK Biobank participants

In this python script, I use a function in sklearn to find the nearest modelled air pollutant value based longitude and latitude values. I then asign this value to each UK Biobank participant in the COVID-19 subset. Source: https://automating-gis-processes.github.io/site/develop/notebooks/L3/nearest-neighbor-faster.html



```python
#build functions
from sklearn.neighbors import BallTree
import numpy as np

def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius

    return closest_points
```


```python
# load AP data
import geopandas as gpd

no2_df = pd.read_csv('data/processed_no2_lonlat.csv', usecols = ['no2_lon','no2_lat','no2_val'])
o3_df = pd.read_csv('data/processed_o3_lonlat.csv', usecols = ['o3_lon','o3_lat','o3_val'])
pm10_df = pd.read_csv('data/processed_pm10_lonlat.csv', usecols = ['pm10_lon','pm10_lat','pm10_val'])
so2_df = pd.read_csv('data/processed_so2_lonlat.csv', usecols = ['so2_lon','so2_lat','so2_val'])
nox_df = pd.read_csv('data/processed_nox_lonlat.csv', usecols = ['nox_lon','nox_lat','nox_val'])
pm25_df = pd.read_csv("data/processed_pm25_lonlat.csv", usecols=['pm25_lon','pm25_lat','pm25_val'])
```


```python
no2_gdf = gpd.GeoDataFrame(
    no2_df, geometry=gpd.points_from_xy(no2_df.no2_lon, no2_df.no2_lat))
nox_gdf = gpd.GeoDataFrame(
    nox_df, geometry=gpd.points_from_xy(nox_df.nox_lon, nox_df.nox_lat))
so2_gdf = gpd.GeoDataFrame(
    so2_df, geometry=gpd.points_from_xy(so2_df.so2_lon, so2_df.so2_lat))
pm25_gdf = gpd.GeoDataFrame(
    pm25_df, geometry=gpd.points_from_xy(pm25_df.pm25_lon, pm25_df.pm25_lat))
pm10_gdf = gpd.GeoDataFrame(
    pm10_df, geometry=gpd.points_from_xy(pm10_df.pm10_lon, pm10_df.pm10_lat))
o3_gdf = gpd.GeoDataFrame(
    o3_df, geometry=gpd.points_from_xy(o3_df.o3_lon, o3_df.o3_lat))
```


```python
ukb_covid_df = pd.read_csv("data_output/ukb_covid_lonlat_df.csv", usecols=['ukb_lon','ukb_lat','eid'])
ukb_covid_gdf = gpd.GeoDataFrame(
    ukb_covid_df, geometry=gpd.points_from_xy(ukb_covid_df.ukb_lon, ukb_covid_df.ukb_lat))

#match no2
no2_matched_df = nearest_neighbor(ukb_covid_gdf, no2_gdf, return_dist=True).rename(columns={'geometry': 'no2_geom'})
#join
no2_ukb_df = ukb_covid_gdf.join(no2_matched_df)[["eid", "no2_val"]]

#match nox
nox_matched_df = nearest_neighbor(ukb_covid_gdf, nox_gdf, return_dist=True).rename(columns={'geometry': 'nox_geom'})
nox_ukb_df = ukb_covid_gdf.join(nox_matched_df)[["eid", "nox_val"]]

#match so2
so2_matched_df = nearest_neighbor(ukb_covid_gdf, so2_gdf, return_dist=True).rename(columns={'geometry': 'so2_geom'})
so2_ukb_df = ukb_covid_gdf.join(so2_matched_df)[["eid", "so2_val"]]

#match pm10
pm10_matched_df = nearest_neighbor(ukb_covid_gdf, pm10_gdf, return_dist=True).rename(columns={'geometry': 'pm10_geom'})
pm10_ukb_df = ukb_covid_gdf.join(pm10_matched_df)[["eid", "pm10_val"]]

#match o3
o3_matched_df = nearest_neighbor(ukb_covid_gdf, o3_gdf, return_dist=True).rename(columns={'geometry': 'o3_geom'})
o3_ukb_df = ukb_covid_gdf.join(o3_matched_df)[["eid", "o3_val"]]

pm25_matched_df = nearest_neighbor(ukb_covid_gdf, pm25_gdf, return_dist=True).rename(columns={'geometry': 'pm25_geom'})
pm25_ukb_df = ukb_covid_gdf.join(pm25_matched_df)[["eid", "pm25_val"]]

```


```python
#merge everything 
from functools import reduce
conc_dfs = [no2_ukb_df,nox_ukb_df,so2_ukb_df,pm10_ukb_df,o3_ukb_df,pm25_ukb_df,]
ukb_covid_AP_vals_merged = reduce(lambda left,right: pd.merge(left,right,on='eid'), conc_dfs)

# ukb_covid_AP_vals_merged.head()
```


```python
ukb_covid_AP_vals_merged.to_csv("data_output/ukb_covid_AP_vals_merged.csv", index = False)
```

### Assign a population density value to each UK Biobank participant

#### Use the modelled N of people per 30m2

For the population density data, I downloaded the most precise and up-to-date models. This data has longitude and latitude values for every 30m2 as well as the modelled population value. In other words, these are estimations of "the number of people living within 30-meter grid tiles in nearly every country around the world." - as stated on their website. 

Citation:<br>
Facebook Connectivity Lab and Center for International Earth Science Information Network - CIESIN - Columbia University. 2016. High Resolution Settlement Layer (HRSL). Source imagery for HRSL © 2016 DigitalGlobe. Accessed 25 JUNE 2020.

Note: This dataset is NOT included in this repository because it is too large. The raw data can be downloaded here: https://data.humdata.org/dataset/united-kingdom-high-resolution-population-density-maps-demographic-estimates


```python
#load - this data is >1GB
pop_dens = pd.read_csv('data/population_gbr_2019-07-01.csv')
```


```python
pop_dens_gdf = gpd.GeoDataFrame(
    pop_dens, geometry=gpd.points_from_xy(pop_dens.Lon, pop_dens.Lat))
pop_dens_matched_df = nearest_neighbor(ukb_covid_gdf, pop_dens_gdf, return_dist=True).rename(columns={'geometry': 'pop_dens_geom'})
#join
pop_dens_joined_df = ukb_covid_gdf.join(pop_dens_matched_df)[["eid", "Population"]]
ukb_covid_AP_popDens_merged = pd.merge(ukb_covid_AP_vals_merged, pop_dens_joined_df, on='eid')
# ukb_covid_AP_popDens_merged.head()
```


```python
ukb_covid_AP_popDens_merged.to_csv("data_output/ukbCov_ukb_covid_AP_popDens_merged.csv", index = False)
```

#### Investigation on movement data

Here, I also include movement data from Facebook: https://research.fb.com/blog/2020/06/protecting-privacy-in-facebook-mobility-data-during-the-covid-19-response/


Columns, from (https://s3.eu-central-1.amazonaws.com/hdx-ckan-filestore-prod/resources/435ed157-6f7a-4e8f-a63a-2aa177b9bd05/readme.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=180&X-Amz-Credential=AKIARZNKTAO7XQY7ED6N%2F20200625%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-SignedHeaders=host&X-Amz-Date=20200625T232711Z&X-Amz-Signature=3d37cad1da6ab0ffbd9d0fa103ccf3afa8998e3004b74b4634b8f0e24a069a0b)

- ds: Date stamp for movement range data row in YYYY-MM-DD form
- country: Three-character ISO-3166 country code
- polygon_source: Source of region polygon, either â€œFIPSâ€ for U.S. data or â€œGADMâ€ for global data
- polygon_id: Unique identifier for region polygon, either numeric string for U.S. FIPS codes or alphanumeric string for GADM regions
- polygon_name: Region name
- all_day_bing_tiles_visited_relative_change: Positive or negative change in movement relative to baseline
- all_day_ratio_single_tile_users: Positive proportion of users staying put within a single location
- baseline_name: When baseline movement was calculated pre-COVID-19
- baseline_type: How baseline movement was calculated pre-COVID-19



```python
fb_mvnt = pd.read_csv('data/movement-range-2020-06-23.txt', sep = "\t")
fb_mvnt_gb = fb_mvnt[fb_mvnt['country'] == "GBR"]
```

    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3063: DtypeWarning: Columns (3) have mixed types.Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
fb_mvnt_gb['polygon_name'].describe()
```




    count                  456
    unique                   4
    top       Northern Ireland
    freq                   114
    Name: polygon_name, dtype: object



This shows that the data is not precise enough. There is only data at the regional level. I am therefore satisfied to only use the well-curated population density data.

#### We will not include the mobility data

## Merge generated data with data from the UK Biobank


```python
ukb_covidAP_merged = pd.merge(pd.read_csv("data_output/merged_ukb_covid_test_results.csv"), 
                              pd.read_csv("data_output/ukbCov_ukb_covid_AP_popDens_merged.csv"), on='eid')
ukb_covidAP_merged.columns
```




    Index(['eid', 'n_cancers', 'townsend', 'x_coord', 'y_coord', 'smoking', 'fev',
           'ethnicity', 'copd', 'WP_dusty', 'WP_chemicals', 'WP_cig', 'WP_diesel',
           'breathing', 'whistling', 'diabetes', 'sex', 'birthYear', 'waist',
           'hip', 'height', 'peopleInHousehold', 'AverageHouseholdIncome',
           'travelToWork', 'result', 'PD_diag', 'AD_diag', 'highBP', 'age',
           'no2_val', 'nox_val', 'so2_val', 'pm10_val', 'o3_val', 'pm25_val',
           'Population'],
          dtype='object')




```python
ukb_covidAP_merged['whr'] = ukb_covidAP_merged['waist']/ukb_covidAP_merged['hip']
ukb_covidAP_merged.to_csv("data_output/ukb_covidAP_merged_full_curatedData.csv")
```

### Drop unused columns


```python
ukb_covidAP = ukb_covidAP_merged.drop([
                       'x_coord','y_coord',
                        'fev','copd',
                        'WP_dusty', 'WP_chemicals',
                        'WP_cig', 'WP_diesel', 'breathing',
                        'birthYear', 
                        'height','travelToWork','waist', 'hip'], axis=1)
ukb_covidAP.to_csv("data_output/ukb_covidAP_finalCuratedData.csv", index = False)
```

### Add 2 more variables to the list: education and care home

I extracted these variables and stored them in a separate folder so that they are not uploaded on github. Access to those data should be done via UK Biobank. 

#### Get the most up-to-date data from these participants


```r
%%R

care_home_edu_dt = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/data/care_home_edu_dt.csv")

#aggregate the columns by selecting the last results, for each 
library(reshape)
care_home_edu_dt_t = melt(care_home_edu_dt, id='eid')

#order by variable to make sure I am will be replacing the latest value 
care_home_edu_dt_t <- care_home_edu_dt_t[order(care_home_edu_dt_t$variable),] 
#delete everything after period 
care_home_edu_dt_t$variable = gsub("\\..*","",care_home_edu_dt_t$variable)

#aggregate by last 
care_home_edu_dt_t_na = na.omit(care_home_edu_dt_t)
care_home_edu_dt_t_na = aggregate(care_home_edu_dt_t_na, by=list(care_home_edu_dt_t_na$eid,care_home_edu_dt_t_na$variable), FUN=tail, n = 1)
# Curate
care_home_edu_dt_t_na = care_home_edu_dt_t_na[c("eid", "variable", "value")]
care_home_edu_dt_t_na$variable = as.factor(care_home_edu_dt_t_na$variable)

# put it back straight 
care_home_edu_cur = cast(care_home_edu_dt_t_na, eid~variable)
head(care_home_edu_cur)
```


          eid X6138 X670
    1 1000013     2    1
    2 1000024    -7    1
    3 1000036     3    1
    4 1000048     6    1
    5 1000055     2    1
    6 1000067     6    1



#### Decode 

Prefer not to answer is considered as NA


```r
%%R
# I keep the ordered function because all of the variables need to be decoded
care_home_edu_dec = care_home_edu_cur
#housing
lvl.100286 <- c(-7,-3,1,2,3,4,5)
lbl.100286 <- c("None of the above","NA","A house or bungalow","A flat, maisonette or apartment","Mobile or temporary structure (i.e. caravan)","Sheltered accommodation","Care home")

care_home_edu_dec$X670 <- ordered(care_home_edu_dec$X670, levels=lvl.100286, labels=lbl.100286)


#education
lvl.100305 <- c(-7,-3,1,2,3,4,5,6)
lbl.100305 <- c("None of the above","NA","College or University degree","A levels/AS levels or equivalent","O levels/GCSEs or equivalent","CSEs or equivalent","NVQ or HND or HNC or equivalent","Other professional qualifications eg: nursing, teaching")
care_home_edu_dec$X6138 <- ordered(care_home_edu_dec$X6138, levels=lvl.100305, labels=lbl.100305)
colnames(care_home_edu_dec) <- c("eid","edu_level","house_type")
write.csv(care_home_edu_dec,"data_output/care_home_edu_decoded.csv", row.names = FALSE)
```


```r
%%R
#Merge to data
temp_merged = merge(read.csv("data_output/ukb_covidAP_finalCuratedData.csv"), read.csv("data_output/care_home_edu_decoded.csv"), by = "eid")
write.csv(temp_merged,"data_output/ukb_covidAP_finalCuratedData_house.csv")
```

### Replace "No Answer" as NA


```r
%%R
ukb_covidAP = read.csv("data_output/ukb_covidAP_finalCuratedData_house.csv")
ukb_covidAP[] <- lapply(ukb_covidAP, function(x) (gsub("Prefer not to answer", "NA", x)))
ukb_covidAP[] <- lapply(ukb_covidAP, function(x) (gsub("Do not know", "NA", x)))

ukb_covidAP$ethnicity_sim <- ifelse(ukb_covidAP$ethnicity == "White" | 
                                ukb_covidAP$ethnicity == "British", "white", "minority")
write.csv(ukb_covidAP,"data_output/ukb_covidAP_finalCuratedData_house.csv", row.names=FALSE)
```

### Add addtional variables: blood type, C-reactive proteins and image derived variable

#### Data curation


```r
%%R

dem_dt = (read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/data/vasc_fronto_all_cause_AD_PD_dt.csv"))

idp_dt = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/data/IDP_subset_dt.csv")

add_dt_merged = merge(dem_dt, idp_dt, by = "eid")
print(nrow(add_dt_merged))
#get the most up-to-date data from all of these variables
library(reshape)
add_dt_merged_t = melt(add_dt_merged, id='eid')

#order by variable to make sure I am will be replacing the latest value 
add_dt_merged_t <- add_dt_merged_t[order(add_dt_merged_t$variable),] 
#delete everything after period 
add_dt_merged_t$variable = gsub("\\..*","",add_dt_merged_t$variable)

#aggregate by last 
add_dt_merged_t = na.omit(add_dt_merged_t)
add_dt_merged_t = aggregate(add_dt_merged_t, by=list(add_dt_merged_t$eid,add_dt_merged_t$variable), FUN=tail, n = 1)
# Curate
add_dt_merged_t = add_dt_merged_t[c("eid", "variable", "value")]
add_dt_merged_t$variable = as.factor(add_dt_merged_t$variable)

# put it back straight 
add_dt_cur = cast(add_dt_merged_t, eid~variable)
nrow(add_dt_cur)
```


    [1] 502505
    [1] 502505



#### Decode


```r
%%R
# I keep the ordered function because all of the variables need to be decoded
add_dt_dec = add_dt_cur
#housing
lvl.0272 <- c(1)
lbl.0272 <- c("Date is unknown")
lvl.0300 <- c(0,1,2)
lbl.0300 <- c("Self-reported only","Hospital admission","Death only")

add_dt_dec$X42018 <- yy_replace(add_dt_dec$X42018, lvl.0272, lbl.0272)
add_dt_dec$X42019 <- yy_replace(add_dt_dec$X42019, lvl.0300, lbl.0300)

add_dt_dec$X42020 <- yy_replace(add_dt_dec$X42020, lvl.0272, lbl.0272)
add_dt_dec$X42021 <- yy_replace(add_dt_dec$X42021, lvl.0300, lbl.0300)

add_dt_dec$X42022 <- yy_replace(add_dt_dec$X42022, lvl.0272, lbl.0272)
add_dt_dec$X42023 <- yy_replace(add_dt_dec$X42023, lvl.0300, lbl.0300)

add_dt_dec$X42024 <- yy_replace(add_dt_dec$X42024, lvl.0272, lbl.0272)
add_dt_dec$X42025 <- yy_replace(add_dt_dec$X42025, lvl.0300, lbl.0300)
```


Add custom colun names for the new variables


```r
%%R

dem_metadt = (read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/vasc_fronto_all_cause_AD_PD.csv"))[c("my_colname","FieldID")]

idp_metadt = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/ ")[c("my_colname","FieldID")]

# dem_metadt$FieldID = toString(dem_metadt$FieldID)
idp_metadt$FieldID = as.integer(gsub("-.*","",idp_metadt$FieldID))


metadt = rbind(idp_metadt,dem_metadt)
metadt$FieldID = paste("X",metadt$FieldID, sep = "")
metadt$my_colname = gsub("_1","",metadt$my_colname)
metadt$my_colname = gsub("_2","",metadt$my_colname)
metadt = metadt[order(metadt$FieldID, decreasing = FALSE), ]

# metadt


```

The columns are in order as shown here, so I can just order the metadt and replace the column names


```r
%%R
colnames(add_dt_dec)
# length(metadt$my_colname)
colnames(add_dt_dec) <- c("eid",unique(metadt$my_colname))
```

#### Curate dementia-related variables


```r
%%R

library(dplyr)
# dementia diagnosis
add_dt_cur = add_dt_dec %>%
  mutate(allDem_diag = c("FALSE", "TRUE")[(!is.na(all_cause_dem_source) 
                                     )+1] )
# Ad diagnosis
add_dt_cur = add_dt_cur %>%
  mutate(AD_diag = c("FALSE", "TRUE")[(!is.na(AD_source)
                                     )+1] )
# fronto_dem_source diagnosis
add_dt_cur = add_dt_cur %>%
  mutate(frontoDem_diag = c("FALSE", "TRUE")[(!is.na(fronto_dem_source)
                                     )+1] )
    
# vasc_dem_source diagnosis
add_dt_cur = add_dt_cur %>%
  mutate(vascDem_diag = c("FALSE", "TRUE")[(!is.na(vasc_dem_source)
                                     )+1] )
# head(add_dt_cur)
```

    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: 
    Attaching package: ‘dplyr’
    
    
      warnings.warn(x, RRuntimeWarning)
    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: The following object is masked from ‘package:reshape’:
    
        rename
    
    
      warnings.warn(x, RRuntimeWarning)
    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    
      warnings.warn(x, RRuntimeWarning)
    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    
    
      warnings.warn(x, RRuntimeWarning)


#### Normalising CRP and WMH


```r
%%R

add_dt_cur$crp_norm = as.numeric(paste(add_dt_cur$crp))/as.numeric(paste(add_dt_cur$tot_p))
add_dt_cur$wmh_norm = as.numeric(paste(add_dt_cur$wmh))/as.numeric(paste(add_dt_cur$brainVol))
```

#### Save


```r
%%R
write.csv(add_dt_cur,"/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/data/curated_idp_dementia_CRP.csv", row.names = FALSE)

```

#### Combine with previous dataset, adding blood types


```r
%%R
new_dt = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/data/curated_idp_dementia_CRP.csv")
new_dt = subset(new_dt, select = -c(AD_diag))
merged_dt = merge(read.csv("data_output/ukb_covidAP_finalCuratedData_house.csv"), 
                 new_dt,
                 by = "eid", all.x = TRUE)

final_merged_dt = merge(merged_dt,read.csv("data/covid19_misc_2_8_2020.txt", sep = "\t"),
                 by = "eid", all.x = TRUE)
write.csv(final_merged_dt, "data_output/ukb_covid_dt_out.csv", row.names=FALSE)


```

# Analysis: COVID-19 cases

## Descriptive statistics - data in models only


```r
%%R
library(arsenal)
ukb_covidAP = read.csv("data_output/ukb_covid_dt_out.csv")
ukb_descript_stats <- tableby(result ~ ., data = ukb_covidAP[,2:ncol(ukb_covidAP)])
write2pdf(ukb_descript_stats, "ukb_descriptStats.pdf",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data") # passed to summary.tableby
write2word(ukb_descript_stats, "ukb_descriptStats.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data") # passed to summary.tableby
```

#### Distributions of each variable


```r
%%R
library(tidyr)
library(purrr)
library(ggplot2)
gen_dt = read.csv("data_output/ukb_covid_dt_out.csv")

cur_dt = subset(gen_dt, select = c(result, Population, n_cancers,townsend, smoking, whistling, diabetes,
                                 whr , sex , age ,peopleInHousehold,AverageHouseholdIncome,ethnicity_sim,highBP,
                          PD_diag, AD_diag , vascDem_diag,frontoDem_diag,
                          greyVol , whiteVol , wmh_norm,
                          edu_score, crp_norm,
                          edu_level , house_type,blood_group,
                          pm25_val,no2_val,nox_val,so2_val,pm10_val,o3_val))
    
```

    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: 
    Attaching package: ‘tidyr’
    
    
      warnings.warn(x, RRuntimeWarning)
    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: The following objects are masked from ‘package:reshape’:
    
        expand, smiths
    
    
      warnings.warn(x, RRuntimeWarning)
    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: Use suppressPackageStartupMessages() to eliminate package startup
    messages
    
      warnings.warn(x, RRuntimeWarning)



```r
%%R

cur_dt %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram()+
    theme_classic()
```

    /Users/yizhouyu/miniconda3/envs/yy_37_env2/lib/python3.7/site-packages/rpy2/rinterface/__init__.py:146: RRuntimeWarning: `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
      warnings.warn(x, RRuntimeWarning)



![png](output_105_1.png)



```r
%%R
cur_dt$log_wmh = -log(cur_dt$wmh_norm)
cur_dt$log_crp = -log(cur_dt$crp_norm)
cur_dt$log_edu = log(cur_dt$edu_score)

cur_dt %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram()+
    theme_classic()

ggsave('fig_out/UKB_all_vars_numeric_distribution.pdf', width = 10, height = 10)
ggsave('fig_out/UKB_all_vars_numeric_distribution.png', width = 10, height = 10)
```

wmh, CRP and education scores need to be logged. 


```r
%%R

gen_dt$log_wmh = log(gen_dt$wmh_norm)
gen_dt$log_crp = log(gen_dt$crp_norm)
gen_dt$log_edu = log(gen_dt$edu_score)
write.csv(gen_dt,"data_output/ukb_covid_dt_out.csv",row.names=FALSE)
```


```r
%%R

cur_dt %>%
  keep(is.factor) %>% 
  gather() %>% 
  ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram(stat="count") + 
    theme_classic()+
    theme(axis.text.x = element_text(angle = 90))

ggsave('fig_out/UKB_all_vars_non-numeric_distribution.pdf', width = 10, height = 10)
ggsave('fig_out/UKB_all_vars_non-numeric_distribution.png', width = 10, height = 10)

```

## Initial model: binomial linear model with all variables, for infection


```r
%%R

gen_dt = read.csv("data_output/ukb_covid_dt_out.csv")
colnames(gen_dt)

cur_dt = subset(gen_dt, select = c(result, Population, n_cancers,townsend, smoking, whistling, diabetes,
                                 whr , sex , age ,peopleInHousehold,AverageHouseholdIncome,ethnicity_sim,highBP,
                          PD_diag, AD_diag , vascDem_diag,frontoDem_diag,
#                           greyVol , whiteVol , wmh_norm,
                          edu_score, log_crp,blood_group,log_edu,
                          edu_level , house_type))
```


```r
%%R
cur_dt_na = na.omit(cur_dt)
summary(ukb_covid.b <-glm(data = cur_dt_na, result ~ Population + n_cancers +townsend + smoking + whistling + diabetes+
                                 whr + sex + age +peopleInHousehold+AverageHouseholdIncome+ethnicity_sim+highBP+
                          PD_diag+ AD_diag + vascDem_diag+frontoDem_diag+
#                            greyVol + whiteVol + wmh_norm+
                          log_edu+ log_crp+blood_group+
                          edu_level + house_type
                          , family = 'binomial'))

```


    
    Call:
    glm(formula = result ~ Population + n_cancers + townsend + smoking + 
        whistling + diabetes + whr + sex + age + peopleInHousehold + 
        AverageHouseholdIncome + ethnicity_sim + highBP + PD_diag + 
        AD_diag + vascDem_diag + frontoDem_diag + log_edu + log_crp + 
        blood_group + edu_level + house_type, family = "binomial", 
        data = cur_dt_na)
    
    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -1.1790  -0.5366  -0.4556  -0.3817   2.5883  
    
    Coefficients:
                                                                       Estimate
    (Intercept)                                                      -6.381e-01
    Population                                                       -3.813e-03
    n_cancers                                                        -2.025e-01
    townsend                                                          2.865e-02
    smokingNever                                                      1.295e-01
    smokingPrevious                                                   1.908e-01
    whistlingTRUE                                                     1.459e-02
    diabetesTRUE                                                      1.997e-02
    whr                                                               5.835e-01
    sexMale                                                           1.503e-01
    age                                                              -3.405e-02
    peopleInHousehold                                                 5.918e-02
    AverageHouseholdIncome18,000 to 30,999                           -1.241e-01
    AverageHouseholdIncome31,000 to 51,999                           -6.947e-02
    AverageHouseholdIncome52,000 to 100,000                          -2.111e-01
    AverageHouseholdIncomeGreater than 100,000                       -4.973e-01
    AverageHouseholdIncomeLess than 18,000                           -5.053e-02
    ethnicity_simwhite                                               -3.192e-01
    highBPFalse                                                       5.085e-02
    highBPTrue                                                        2.302e-02
    PD_diagTrue                                                       6.663e-01
    AD_diagTrue                                                       1.521e+00
    vascDem_diagTRUE                                                  1.498e+00
    frontoDem_diagTRUE                                                1.705e+01
    log_edu                                                           9.257e-02
    log_crp                                                           3.459e-03
    blood_groupAB                                                    -8.089e-02
    blood_groupAO                                                    -2.660e-02
    blood_groupBB                                                    -1.843e-01
    blood_groupBO                                                     5.725e-02
    blood_groupOO                                                    -5.122e-02
    edu_levelCollege or University degree                             2.570e-01
    edu_levelCSEs or equivalent                                       2.349e-01
    edu_levelNone of the above                                        4.750e-01
    edu_levelNVQ or HND or HNC or equivalent                          3.046e-01
    edu_levelO levels/GCSEs or equivalent                             6.255e-02
    edu_levelOther professional qualifications eg: nursing, teaching  3.150e-01
    house_typeA house or bungalow                                     3.317e-02
    house_typeCare home                                              -1.310e+01
    house_typeMobile or temporary structure (i.e. caravan)           -1.239e+01
    house_typeNone of the above                                      -5.282e-01
    house_typeSheltered accommodation                                -1.251e+01
                                                                     Std. Error
    (Intercept)                                                       1.017e+03
    Population                                                        2.343e-03
    n_cancers                                                         1.087e-01
    townsend                                                          1.320e-02
    smokingNever                                                      1.127e-01
    smokingPrevious                                                   1.142e-01
    whistlingTRUE                                                     7.958e-02
    diabetesTRUE                                                      1.184e-01
    whr                                                               5.243e-01
    sexMale                                                           9.152e-02
    age                                                               4.813e-03
    peopleInHousehold                                                 2.167e-02
    AverageHouseholdIncome18,000 to 30,999                            1.017e+03
    AverageHouseholdIncome31,000 to 51,999                            1.017e+03
    AverageHouseholdIncome52,000 to 100,000                           1.017e+03
    AverageHouseholdIncomeGreater than 100,000                        1.017e+03
    AverageHouseholdIncomeLess than 18,000                            1.017e+03
    ethnicity_simwhite                                                9.838e-02
    highBPFalse                                                       2.406e-01
    highBPTrue                                                        2.420e-01
    PD_diagTrue                                                       2.702e-01
    AD_diagTrue                                                       3.271e-01
    vascDem_diagTRUE                                                  4.343e-01
    frontoDem_diagTRUE                                                8.827e+02
    log_edu                                                           3.517e-02
    log_crp                                                           3.388e-02
    blood_groupAB                                                     2.205e-01
    blood_groupAO                                                     1.313e-01
    blood_groupBB                                                     4.587e-01
    blood_groupBO                                                     1.587e-01
    blood_groupOO                                                     1.298e-01
    edu_levelCollege or University degree                             2.436e-01
    edu_levelCSEs or equivalent                                       2.585e-01
    edu_levelNone of the above                                        2.362e-01
    edu_levelNVQ or HND or HNC or equivalent                          2.371e-01
    edu_levelO levels/GCSEs or equivalent                             2.388e-01
    edu_levelOther professional qualifications eg: nursing, teaching  2.296e-01
    house_typeA house or bungalow                                     1.181e-01
    house_typeCare home                                               8.827e+02
    house_typeMobile or temporary structure (i.e. caravan)            2.428e+02
    house_typeNone of the above                                       7.531e-01
    house_typeSheltered accommodation                                 5.059e+02
                                                                     z value
    (Intercept)                                                       -0.001
    Population                                                        -1.627
    n_cancers                                                         -1.864
    townsend                                                           2.170
    smokingNever                                                       1.149
    smokingPrevious                                                    1.671
    whistlingTRUE                                                      0.183
    diabetesTRUE                                                       0.169
    whr                                                                1.113
    sexMale                                                            1.643
    age                                                               -7.075
    peopleInHousehold                                                  2.732
    AverageHouseholdIncome18,000 to 30,999                             0.000
    AverageHouseholdIncome31,000 to 51,999                             0.000
    AverageHouseholdIncome52,000 to 100,000                            0.000
    AverageHouseholdIncomeGreater than 100,000                         0.000
    AverageHouseholdIncomeLess than 18,000                             0.000
    ethnicity_simwhite                                                -3.244
    highBPFalse                                                        0.211
    highBPTrue                                                         0.095
    PD_diagTrue                                                        2.466
    AD_diagTrue                                                        4.651
    vascDem_diagTRUE                                                   3.450
    frontoDem_diagTRUE                                                 0.019
    log_edu                                                            2.632
    log_crp                                                            0.102
    blood_groupAB                                                     -0.367
    blood_groupAO                                                     -0.203
    blood_groupBB                                                     -0.402
    blood_groupBO                                                      0.361
    blood_groupOO                                                     -0.394
    edu_levelCollege or University degree                              1.055
    edu_levelCSEs or equivalent                                        0.909
    edu_levelNone of the above                                         2.011
    edu_levelNVQ or HND or HNC or equivalent                           1.285
    edu_levelO levels/GCSEs or equivalent                              0.262
    edu_levelOther professional qualifications eg: nursing, teaching   1.372
    house_typeA house or bungalow                                      0.281
    house_typeCare home                                               -0.015
    house_typeMobile or temporary structure (i.e. caravan)            -0.051
    house_typeNone of the above                                       -0.701
    house_typeSheltered accommodation                                 -0.025
                                                                     Pr(>|z|)    
    (Intercept)                                                       0.99950    
    Population                                                        0.10365    
    n_cancers                                                         0.06238 .  
    townsend                                                          0.03001 *  
    smokingNever                                                      0.25067    
    smokingPrevious                                                   0.09471 .  
    whistlingTRUE                                                     0.85453    
    diabetesTRUE                                                      0.86600    
    whr                                                               0.26577    
    sexMale                                                           0.10044    
    age                                                              1.50e-12 ***
    peopleInHousehold                                                 0.00630 ** 
    AverageHouseholdIncome18,000 to 30,999                            0.99990    
    AverageHouseholdIncome31,000 to 51,999                            0.99995    
    AverageHouseholdIncome52,000 to 100,000                           0.99983    
    AverageHouseholdIncomeGreater than 100,000                        0.99961    
    AverageHouseholdIncomeLess than 18,000                            0.99996    
    ethnicity_simwhite                                                0.00118 ** 
    highBPFalse                                                       0.83261    
    highBPTrue                                                        0.92421    
    PD_diagTrue                                                       0.01368 *  
    AD_diagTrue                                                      3.31e-06 ***
    vascDem_diagTRUE                                                  0.00056 ***
    frontoDem_diagTRUE                                                0.98459    
    log_edu                                                           0.00848 ** 
    log_crp                                                           0.91868    
    blood_groupAB                                                     0.71374    
    blood_groupAO                                                     0.83948    
    blood_groupBB                                                     0.68789    
    blood_groupBO                                                     0.71836    
    blood_groupOO                                                     0.69323    
    edu_levelCollege or University degree                             0.29151    
    edu_levelCSEs or equivalent                                       0.36356    
    edu_levelNone of the above                                        0.04436 *  
    edu_levelNVQ or HND or HNC or equivalent                          0.19883    
    edu_levelO levels/GCSEs or equivalent                             0.79340    
    edu_levelOther professional qualifications eg: nursing, teaching  0.17014    
    house_typeA house or bungalow                                     0.77887    
    house_typeCare home                                               0.98816    
    house_typeMobile or temporary structure (i.e. caravan)            0.95931    
    house_typeNone of the above                                       0.48307    
    house_typeSheltered accommodation                                 0.98028    
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for binomial family taken to be 1)
    
        Null deviance: 6421.9  on 8799  degrees of freedom
    Residual deviance: 6205.7  on 8758  degrees of freedom
    AIC: 6289.7
    
    Number of Fisher Scoring iterations: 13
    



### Model analysis 

This model shows that health-related factors, like waist-hig ratios (whr), blood pressure (BP), previous history of cancers, sex and diabetes are not predictors of infections. This makes sense since, technically, people with existing cardiovascular and health issues are equally at risk of getting the disease.<br>
The Townsend index, PM2.5, number of people in a house hold, Alzheimer's disease disgnosis and age are predictors of infections. Possible explanations could be: 
- townsend: financially "deprived" people are likely to get infected;
- PM2.5: PM might act as a "carpet" that facilitates the retention and survival of viruses in the air;
- People in the household: More people in a household would increase the risk of transmission exponentially;
- AD: People with dementia might be living in care homes, and the virus would be transmitted by carers. 
- Age is a negative predictor: younger healthy people might be more socially active, thus have increased risk of getting the disease.

Here, I will create a model without the house_type variable. The reason for including this variable was to accound for whether people in care homes are more likely to get infected. However, only a few individuals in this cohort report being in a care home. This variable is therefore not relevant anymore. 


```r
%%R
stargazer::stargazer(ukb_covid.b, type="html",out = "fig_out/ukb_covid_full_model_binary.html",
          dep.var.labels="COVID positive or not",
          single.row=TRUE)
```


    
    <table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td><em>Dependent variable:</em></td></tr>
    <tr><td></td><td colspan="1" style="border-bottom: 1px solid black"></td></tr>
    <tr><td style="text-align:left"></td><td>COVID positive or not</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Population</td><td>-0.004 (0.002)</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>-0.202<sup>*</sup> (0.109)</td></tr>
    <tr><td style="text-align:left">townsend</td><td>0.029<sup>**</sup> (0.013)</td></tr>
    <tr><td style="text-align:left">smokingNever</td><td>0.129 (0.113)</td></tr>
    <tr><td style="text-align:left">smokingPrevious</td><td>0.191<sup>*</sup> (0.114)</td></tr>
    <tr><td style="text-align:left">whistling</td><td>0.015 (0.080)</td></tr>
    <tr><td style="text-align:left">diabetes</td><td>0.020 (0.118)</td></tr>
    <tr><td style="text-align:left">whr</td><td>0.583 (0.524)</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>0.150 (0.092)</td></tr>
    <tr><td style="text-align:left">age</td><td>-0.034<sup>***</sup> (0.005)</td></tr>
    <tr><td style="text-align:left">peopleInHousehold</td><td>0.059<sup>***</sup> (0.022)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome18,000 to 30,999</td><td>-0.124 (1,017.458)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome31,000 to 51,999</td><td>-0.069 (1,017.458)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome52,000 to 100,000</td><td>-0.211 (1,017.458)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeGreater than 100,000</td><td>-0.497 (1,017.458)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeLess than 18,000</td><td>-0.051 (1,017.458)</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>-0.319<sup>***</sup> (0.098)</td></tr>
    <tr><td style="text-align:left">highBPFalse</td><td>0.051 (0.241)</td></tr>
    <tr><td style="text-align:left">highBPTrue</td><td>0.023 (0.242)</td></tr>
    <tr><td style="text-align:left">PD_diagTrue</td><td>0.666<sup>**</sup> (0.270)</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>1.521<sup>***</sup> (0.327)</td></tr>
    <tr><td style="text-align:left">vascDem_diag</td><td>1.498<sup>***</sup> (0.434)</td></tr>
    <tr><td style="text-align:left">frontoDem_diag</td><td>17.051 (882.743)</td></tr>
    <tr><td style="text-align:left">log_edu</td><td>0.093<sup>***</sup> (0.035)</td></tr>
    <tr><td style="text-align:left">log_crp</td><td>0.003 (0.034)</td></tr>
    <tr><td style="text-align:left">blood_groupAB</td><td>-0.081 (0.221)</td></tr>
    <tr><td style="text-align:left">blood_groupAO</td><td>-0.027 (0.131)</td></tr>
    <tr><td style="text-align:left">blood_groupBB</td><td>-0.184 (0.459)</td></tr>
    <tr><td style="text-align:left">blood_groupBO</td><td>0.057 (0.159)</td></tr>
    <tr><td style="text-align:left">blood_groupOO</td><td>-0.051 (0.130)</td></tr>
    <tr><td style="text-align:left">edu_levelCollege or University degree</td><td>0.257 (0.244)</td></tr>
    <tr><td style="text-align:left">edu_levelCSEs or equivalent</td><td>0.235 (0.258)</td></tr>
    <tr><td style="text-align:left">edu_levelNone of the above</td><td>0.475<sup>**</sup> (0.236)</td></tr>
    <tr><td style="text-align:left">edu_levelNVQ or HND or HNC or equivalent</td><td>0.305 (0.237)</td></tr>
    <tr><td style="text-align:left">edu_levelO levels/GCSEs or equivalent</td><td>0.063 (0.239)</td></tr>
    <tr><td style="text-align:left">edu_levelOther professional qualifications eg: nursing, teaching</td><td>0.315 (0.230)</td></tr>
    <tr><td style="text-align:left">house_typeA house or bungalow</td><td>0.033 (0.118)</td></tr>
    <tr><td style="text-align:left">house_typeCare home</td><td>-13.096 (882.743)</td></tr>
    <tr><td style="text-align:left">house_typeMobile or temporary structure (i.e. caravan)</td><td>-12.389 (242.809)</td></tr>
    <tr><td style="text-align:left">house_typeNone of the above</td><td>-0.528 (0.753)</td></tr>
    <tr><td style="text-align:left">house_typeSheltered accommodation</td><td>-12.505 (505.950)</td></tr>
    <tr><td style="text-align:left">Constant</td><td>-0.638 (1,017.458)</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>8,800</td></tr>
    <tr><td style="text-align:left">Log Likelihood</td><td>-3,102.836</td></tr>
    <tr><td style="text-align:left">Akaike Inf. Crit.</td><td>6,289.672</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
    </table>




```r
%%R
## odds ratios and 95% CI

ukb_covid.b_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid.b), 
                           conf_int_low = summary(ukb_covid.b)$coefficients[,1] - 
                                          summary(ukb_covid.b)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid.b)$coefficients[,1] + 
                                          summary(ukb_covid.b)$coefficients[,2])),
                 p_value = summary(ukb_covid.b)$coefficients[,4]))

stargazer::stargazer(ukb_covid.b_odds, type ="html", 
                     single.row=TRUE, summary = FALSE, 
                     out = "fig_out/ukb_covid_odds_fullmodel.html")
write.csv(ukb_covid.b_odds,"data_output/ukb_covid_full_model_odds.csv", row.names=TRUE)

ukb_covid.b_odds$significance = "p-value > 0.05"
ukb_covid.b_odds$significance[ukb_covid.b_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid.b_odds$variables = row.names(ukb_covid.b_odds)

library(ggplot2)
ggplot(ukb_covid.b_odds[which(ukb_covid.b_odds$significance == "p-value < 0.05"),], 
       aes(x=reorder(variables, OR), y=OR)) + 
    geom_point(shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Infectivity odds ratios") + 
  xlab("Variables")
ggsave('fig_out/UKB_covidInfection_full_model_ORplot_sig_only.pdf', width = 6, height = 4)
ggsave('fig_out/UKB_covidInfection_full_model_ORplot_sig_only.png', width = 6, height = 4)

```


    
    <table style="text-align:center"><tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td>OR</td><td>conf_int_low</td><td>conf_int_high</td><td>p_value</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">(Intercept)</td><td>0.528</td><td>0</td><td>Inf</td><td>0.999</td></tr>
    <tr><td style="text-align:left">Population</td><td>0.996</td><td>0.994</td><td>0.999</td><td>0.104</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>0.817</td><td>0.733</td><td>0.910</td><td>0.062</td></tr>
    <tr><td style="text-align:left">townsend</td><td>1.029</td><td>1.016</td><td>1.043</td><td>0.030</td></tr>
    <tr><td style="text-align:left">smokingNever</td><td>1.138</td><td>1.017</td><td>1.274</td><td>0.251</td></tr>
    <tr><td style="text-align:left">smokingPrevious</td><td>1.210</td><td>1.080</td><td>1.357</td><td>0.095</td></tr>
    <tr><td style="text-align:left">whistlingTRUE</td><td>1.015</td><td>0.937</td><td>1.099</td><td>0.855</td></tr>
    <tr><td style="text-align:left">diabetesTRUE</td><td>1.020</td><td>0.906</td><td>1.148</td><td>0.866</td></tr>
    <tr><td style="text-align:left">whr</td><td>1.792</td><td>1.061</td><td>3.028</td><td>0.266</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>1.162</td><td>1.061</td><td>1.274</td><td>0.100</td></tr>
    <tr><td style="text-align:left">age</td><td>0.967</td><td>0.962</td><td>0.971</td><td>0</td></tr>
    <tr><td style="text-align:left">peopleInHousehold</td><td>1.061</td><td>1.038</td><td>1.084</td><td>0.006</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome18,000 to 30,999</td><td>0.883</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome31,000 to 51,999</td><td>0.933</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome52,000 to 100,000</td><td>0.810</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeGreater than 100,000</td><td>0.608</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeLess than 18,000</td><td>0.951</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>0.727</td><td>0.659</td><td>0.802</td><td>0.001</td></tr>
    <tr><td style="text-align:left">highBPFalse</td><td>1.052</td><td>0.827</td><td>1.338</td><td>0.833</td></tr>
    <tr><td style="text-align:left">highBPTrue</td><td>1.023</td><td>0.803</td><td>1.303</td><td>0.924</td></tr>
    <tr><td style="text-align:left">PD_diagTrue</td><td>1.947</td><td>1.486</td><td>2.551</td><td>0.014</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>4.578</td><td>3.301</td><td>6.350</td><td>0.00000</td></tr>
    <tr><td style="text-align:left">vascDem_diagTRUE</td><td>4.475</td><td>2.898</td><td>6.909</td><td>0.001</td></tr>
    <tr><td style="text-align:left">frontoDem_diagTRUE</td><td>25,406,220.000</td><td>0</td><td>Inf</td><td>0.985</td></tr>
    <tr><td style="text-align:left">log_edu</td><td>1.097</td><td>1.059</td><td>1.136</td><td>0.008</td></tr>
    <tr><td style="text-align:left">log_crp</td><td>1.003</td><td>0.970</td><td>1.038</td><td>0.919</td></tr>
    <tr><td style="text-align:left">blood_groupAB</td><td>0.922</td><td>0.740</td><td>1.150</td><td>0.714</td></tr>
    <tr><td style="text-align:left">blood_groupAO</td><td>0.974</td><td>0.854</td><td>1.110</td><td>0.839</td></tr>
    <tr><td style="text-align:left">blood_groupBB</td><td>0.832</td><td>0.526</td><td>1.316</td><td>0.688</td></tr>
    <tr><td style="text-align:left">blood_groupBO</td><td>1.059</td><td>0.903</td><td>1.241</td><td>0.718</td></tr>
    <tr><td style="text-align:left">blood_groupOO</td><td>0.950</td><td>0.834</td><td>1.082</td><td>0.693</td></tr>
    <tr><td style="text-align:left">edu_levelCollege or University degree</td><td>1.293</td><td>1.013</td><td>1.650</td><td>0.292</td></tr>
    <tr><td style="text-align:left">edu_levelCSEs or equivalent</td><td>1.265</td><td>0.977</td><td>1.638</td><td>0.364</td></tr>
    <tr><td style="text-align:left">edu_levelNone of the above</td><td>1.608</td><td>1.270</td><td>2.037</td><td>0.044</td></tr>
    <tr><td style="text-align:left">edu_levelNVQ or HND or HNC or equivalent</td><td>1.356</td><td>1.070</td><td>1.719</td><td>0.199</td></tr>
    <tr><td style="text-align:left">edu_levelO levels/GCSEs or equivalent</td><td>1.065</td><td>0.838</td><td>1.352</td><td>0.793</td></tr>
    <tr><td style="text-align:left">edu_levelOther professional qualifications eg: nursing, teaching</td><td>1.370</td><td>1.089</td><td>1.724</td><td>0.170</td></tr>
    <tr><td style="text-align:left">house_typeA house or bungalow</td><td>1.034</td><td>0.919</td><td>1.163</td><td>0.779</td></tr>
    <tr><td style="text-align:left">house_typeCare home</td><td>0.00000</td><td>0</td><td>Inf</td><td>0.988</td></tr>
    <tr><td style="text-align:left">house_typeMobile or temporary structure (i.e. caravan)</td><td>0.00000</td><td>0</td><td>11,761,589,003,334,755,352,685,426,304,803,959,958,831,693,829,258,798,628,853,146,637,434,447,308,612,822,624,784,022,818,131,017,728</td><td>0.959</td></tr>
    <tr><td style="text-align:left">house_typeNone of the above</td><td>0.590</td><td>0.278</td><td>1.252</td><td>0.483</td></tr>
    <tr><td style="text-align:left">house_typeSheltered accommodation</td><td>0.00000</td><td>0</td><td>1</td><td>0.980</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr></table>



### Simplification of the full model




```r
%%R
library(MASS)
# Stepwise regression model
ukb_covid.b_step <- stepAIC(ukb_covid.b, direction = "both")
summary(ukb_covid.b_step)
```


    Start:  AIC=6289.67
    result ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + AverageHouseholdIncome + 
        ethnicity_sim + highBP + PD_diag + AD_diag + vascDem_diag + 
        frontoDem_diag + log_edu + log_crp + blood_group + edu_level + 
        house_type
    
                             Df Deviance    AIC
    - blood_group             5   6206.8 6280.8
    - house_type              5   6210.4 6284.4
    - highBP                  2   6205.9 6285.9
    - AverageHouseholdIncome  5   6212.4 6286.4
    - log_crp                 1   6205.7 6287.7
    - diabetes                1   6205.7 6287.7
    - whistling               1   6205.7 6287.7
    - smoking                 2   6208.6 6288.6
    - whr                     1   6206.9 6288.9
    <none>                        6205.7 6289.7
    - sex                     1   6208.4 6290.4
    - Population              1   6209.1 6291.1
    - n_cancers               1   6209.4 6291.4
    - edu_level               6   6219.6 6291.6
    - townsend                1   6210.4 6292.4
    - frontoDem_diag          1   6210.8 6292.8
    - PD_diag                 1   6211.0 6293.0
    - peopleInHousehold       1   6212.3 6294.3
    - log_edu                 1   6212.8 6294.8
    - vascDem_diag            1   6215.6 6297.6
    - ethnicity_sim           1   6215.8 6297.8
    - AD_diag                 1   6223.3 6305.3
    - age                     1   6255.6 6337.6
    
    Step:  AIC=6280.8
    result ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + AverageHouseholdIncome + 
        ethnicity_sim + highBP + PD_diag + AD_diag + vascDem_diag + 
        frontoDem_diag + log_edu + log_crp + edu_level + house_type
    
                             Df Deviance    AIC
    - house_type              5   6211.6 6275.6
    - highBP                  2   6207.0 6277.0
    - AverageHouseholdIncome  5   6213.6 6277.6
    - log_crp                 1   6206.8 6278.8
    - diabetes                1   6206.8 6278.8
    - whistling               1   6206.8 6278.8
    - smoking                 2   6209.7 6279.7
    - whr                     1   6208.0 6280.0
    <none>                        6206.8 6280.8
    - sex                     1   6209.5 6281.5
    - Population              1   6210.3 6282.3
    - n_cancers               1   6210.5 6282.5
    - edu_level               6   6220.8 6282.8
    - townsend                1   6211.5 6283.5
    - frontoDem_diag          1   6211.9 6283.9
    - PD_diag                 1   6212.2 6284.2
    - peopleInHousehold       1   6213.7 6285.7
    - log_edu                 1   6213.9 6285.9
    - vascDem_diag            1   6216.7 6288.7
    - ethnicity_sim           1   6217.4 6289.4
    + blood_group             5   6205.7 6289.7
    - AD_diag                 1   6224.4 6296.4
    - age                     1   6256.6 6328.6
    
    Step:  AIC=6275.61
    result ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + AverageHouseholdIncome + 
        ethnicity_sim + highBP + PD_diag + AD_diag + vascDem_diag + 
        frontoDem_diag + log_edu + log_crp + edu_level
    
                             Df Deviance    AIC
    - highBP                  2   6211.8 6271.8
    - AverageHouseholdIncome  5   6218.3 6272.3
    - log_crp                 1   6211.6 6273.6
    - diabetes                1   6211.6 6273.6
    - whistling               1   6211.6 6273.6
    - smoking                 2   6214.6 6274.6
    - whr                     1   6212.8 6274.8
    <none>                        6211.6 6275.6
    - sex                     1   6214.3 6276.3
    - Population              1   6215.2 6277.2
    - n_cancers               1   6215.3 6277.3
    - edu_level               6   6225.6 6277.6
    - townsend                1   6216.7 6278.7
    - frontoDem_diag          1   6216.7 6278.7
    - PD_diag                 1   6217.1 6279.1
    + house_type              5   6206.8 6280.8
    - peopleInHousehold       1   6219.0 6281.0
    - log_edu                 1   6219.2 6281.2
    - vascDem_diag            1   6221.5 6283.5
    - ethnicity_sim           1   6222.0 6284.0
    + blood_group             5   6210.4 6284.4
    - AD_diag                 1   6229.3 6291.3
    - age                     1   6261.3 6323.3
    
    Step:  AIC=6271.79
    result ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + AverageHouseholdIncome + 
        ethnicity_sim + PD_diag + AD_diag + vascDem_diag + frontoDem_diag + 
        log_edu + log_crp + edu_level
    
                             Df Deviance    AIC
    - AverageHouseholdIncome  5   6218.5 6268.5
    - log_crp                 1   6211.8 6269.8
    - whistling               1   6211.8 6269.8
    - diabetes                1   6211.8 6269.8
    - smoking                 2   6214.7 6270.7
    - whr                     1   6213.0 6271.0
    <none>                        6211.8 6271.8
    - sex                     1   6214.4 6272.4
    - Population              1   6215.3 6273.3
    - n_cancers               1   6215.5 6273.5
    - edu_level               6   6225.8 6273.8
    - frontoDem_diag          1   6216.9 6274.9
    - townsend                1   6217.0 6275.0
    - PD_diag                 1   6217.3 6275.3
    + highBP                  2   6211.6 6275.6
    + house_type              5   6207.0 6277.0
    - peopleInHousehold       1   6219.3 6277.3
    - log_edu                 1   6219.3 6277.3
    - vascDem_diag            1   6221.7 6279.7
    - ethnicity_sim           1   6222.2 6280.2
    + blood_group             5   6210.6 6280.6
    - AD_diag                 1   6229.5 6287.5
    - age                     1   6263.3 6321.3
    
    Step:  AIC=6268.47
    result ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + ethnicity_sim + 
        PD_diag + AD_diag + vascDem_diag + frontoDem_diag + log_edu + 
        log_crp + edu_level
    
                             Df Deviance    AIC
    - log_crp                 1   6218.5 6266.5
    - whistling               1   6218.5 6266.5
    - diabetes                1   6218.5 6266.5
    - smoking                 2   6221.2 6267.2
    - whr                     1   6219.9 6267.9
    <none>                        6218.5 6268.5
    - sex                     1   6220.5 6268.5
    - Population              1   6221.9 6269.9
    - n_cancers               1   6222.2 6270.2
    - edu_level               6   6233.3 6271.3
    - frontoDem_diag          1   6223.8 6271.8
    + AverageHouseholdIncome  5   6211.8 6271.8
    - PD_diag                 1   6224.0 6272.0
    - townsend                1   6224.1 6272.1
    + highBP                  2   6218.3 6272.3
    - peopleInHousehold       1   6224.6 6272.6
    + house_type              5   6213.7 6273.7
    - vascDem_diag            1   6228.3 6276.3
    + blood_group             5   6217.2 6277.2
    - ethnicity_sim           1   6229.4 6277.4
    - log_edu                 1   6230.1 6278.1
    - AD_diag                 1   6236.5 6284.5
    - age                     1   6268.0 6316.0
    
    Step:  AIC=6266.49
    result ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + ethnicity_sim + 
        PD_diag + AD_diag + vascDem_diag + frontoDem_diag + log_edu + 
        edu_level
    
                             Df Deviance    AIC
    - whistling               1   6218.5 6264.5
    - diabetes                1   6218.5 6264.5
    - smoking                 2   6221.2 6265.2
    - whr                     1   6220.1 6266.1
    <none>                        6218.5 6266.5
    - sex                     1   6220.5 6266.5
    - Population              1   6221.9 6267.9
    - n_cancers               1   6222.2 6268.2
    + log_crp                 1   6218.5 6268.5
    - edu_level               6   6233.4 6269.4
    - frontoDem_diag          1   6223.8 6269.8
    + AverageHouseholdIncome  5   6211.8 6269.8
    - PD_diag                 1   6224.0 6270.0
    - townsend                1   6224.1 6270.1
    + highBP                  2   6218.4 6270.4
    - peopleInHousehold       1   6224.6 6270.6
    + house_type              5   6213.7 6271.7
    - vascDem_diag            1   6228.3 6274.3
    + blood_group             5   6217.2 6275.2
    - ethnicity_sim           1   6229.4 6275.4
    - log_edu                 1   6230.2 6276.2
    - AD_diag                 1   6236.5 6282.5
    - age                     1   6268.1 6314.1
    
    Step:  AIC=6264.52
    result ~ Population + n_cancers + townsend + smoking + diabetes + 
        whr + sex + age + peopleInHousehold + ethnicity_sim + PD_diag + 
        AD_diag + vascDem_diag + frontoDem_diag + log_edu + edu_level
    
                             Df Deviance    AIC
    - diabetes                1   6218.6 6262.6
    - smoking                 2   6221.2 6263.2
    - whr                     1   6220.2 6264.2
    <none>                        6218.5 6264.5
    - sex                     1   6220.5 6264.5
    - Population              1   6222.0 6266.0
    - n_cancers               1   6222.2 6266.2
    + whistling               1   6218.5 6266.5
    + log_crp                 1   6218.5 6266.5
    - edu_level               6   6233.5 6267.5
    - frontoDem_diag          1   6223.8 6267.8
    + AverageHouseholdIncome  5   6211.8 6267.8
    - PD_diag                 1   6224.0 6268.0
    - townsend                1   6224.2 6268.2
    + highBP                  2   6218.4 6268.4
    - peopleInHousehold       1   6224.6 6268.6
    + house_type              5   6213.8 6269.8
    - vascDem_diag            1   6228.3 6272.3
    + blood_group             5   6217.3 6273.3
    - ethnicity_sim           1   6229.4 6273.4
    - log_edu                 1   6230.3 6274.3
    - AD_diag                 1   6236.6 6280.6
    - age                     1   6268.1 6312.1
    
    Step:  AIC=6262.57
    result ~ Population + n_cancers + townsend + smoking + whr + 
        sex + age + peopleInHousehold + ethnicity_sim + PD_diag + 
        AD_diag + vascDem_diag + frontoDem_diag + log_edu + edu_level
    
                             Df Deviance    AIC
    - smoking                 2   6221.3 6261.3
    - whr                     1   6220.5 6262.5
    - sex                     1   6220.5 6262.5
    <none>                        6218.6 6262.6
    - Population              1   6222.0 6264.0
    - n_cancers               1   6222.3 6264.3
    + diabetes                1   6218.5 6264.5
    + whistling               1   6218.5 6264.5
    + log_crp                 1   6218.5 6264.5
    - edu_level               6   6233.6 6265.6
    + AverageHouseholdIncome  5   6211.8 6265.8
    - frontoDem_diag          1   6223.9 6265.9
    - PD_diag                 1   6224.0 6266.0
    - townsend                1   6224.3 6266.3
    + highBP                  2   6218.4 6266.4
    - peopleInHousehold       1   6224.6 6266.6
    + house_type              5   6213.8 6267.8
    - vascDem_diag            1   6228.5 6270.5
    + blood_group             5   6217.3 6271.3
    - ethnicity_sim           1   6229.5 6271.5
    - log_edu                 1   6230.4 6272.4
    - AD_diag                 1   6236.7 6278.7
    - age                     1   6268.2 6310.2
    
    Step:  AIC=6261.32
    result ~ Population + n_cancers + townsend + whr + sex + age + 
        peopleInHousehold + ethnicity_sim + PD_diag + AD_diag + vascDem_diag + 
        frontoDem_diag + log_edu + edu_level
    
                             Df Deviance    AIC
    - sex                     1   6223.3 6261.3
    <none>                        6221.3 6261.3
    - whr                     1   6223.3 6261.3
    + smoking                 2   6218.6 6262.6
    - Population              1   6224.7 6262.7
    - n_cancers               1   6224.9 6262.9
    + diabetes                1   6221.2 6263.2
    + log_crp                 1   6221.3 6263.3
    + whistling               1   6221.3 6263.3
    - edu_level               6   6236.3 6264.3
    - townsend                1   6226.6 6264.6
    - frontoDem_diag          1   6226.7 6264.7
    - PD_diag                 1   6226.7 6264.7
    + AverageHouseholdIncome  5   6214.8 6264.8
    + highBP                  2   6221.2 6265.2
    - peopleInHousehold       1   6227.7 6265.7
    + house_type              5   6216.5 6266.5
    - vascDem_diag            1   6231.2 6269.2
    + blood_group             5   6220.1 6270.1
    - ethnicity_sim           1   6232.3 6270.3
    - log_edu                 1   6232.8 6270.8
    - AD_diag                 1   6239.3 6277.3
    - age                     1   6269.0 6307.0
    
    Step:  AIC=6261.27
    result ~ Population + n_cancers + townsend + whr + age + peopleInHousehold + 
        ethnicity_sim + PD_diag + AD_diag + vascDem_diag + frontoDem_diag + 
        log_edu + edu_level
    
                             Df Deviance    AIC
    <none>                        6223.3 6261.3
    + sex                     1   6221.3 6261.3
    + smoking                 2   6220.5 6262.5
    - Population              1   6226.6 6262.6
    - n_cancers               1   6227.0 6263.0
    + log_crp                 1   6223.2 6263.2
    + diabetes                1   6223.2 6263.2
    + whistling               1   6223.3 6263.3
    - edu_level               6   6238.0 6264.0
    - townsend                1   6228.3 6264.3
    - frontoDem_diag          1   6228.7 6264.7
    - PD_diag                 1   6228.9 6264.9
    + highBP                  2   6223.2 6265.2
    + AverageHouseholdIncome  5   6217.4 6265.4
    - peopleInHousehold       1   6229.9 6265.9
    + house_type              5   6218.5 6266.5
    - whr                     1   6232.2 6268.2
    - vascDem_diag            1   6233.1 6269.1
    - ethnicity_sim           1   6233.8 6269.8
    + blood_group             5   6222.0 6270.0
    - log_edu                 1   6234.3 6270.3
    - AD_diag                 1   6241.2 6277.2
    - age                     1   6270.9 6306.9
    
    Call:
    glm(formula = result ~ Population + n_cancers + townsend + whr + 
        age + peopleInHousehold + ethnicity_sim + PD_diag + AD_diag + 
        vascDem_diag + frontoDem_diag + log_edu + edu_level, family = "binomial", 
        data = cur_dt_na)
    
    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -1.2324  -0.5385  -0.4593  -0.3854   2.5955  
    
    Coefficients:
                                                                       Estimate
    (Intercept)                                                       -1.241031
    Population                                                        -0.003740
    n_cancers                                                         -0.204119
    townsend                                                           0.026925
    whr                                                                1.118596
    age                                                               -0.031004
    peopleInHousehold                                                  0.057422
    ethnicity_simwhite                                                -0.317547
    PD_diagTrue                                                        0.681123
    AD_diagTrue                                                        1.530243
    vascDem_diagTRUE                                                   1.473491
    frontoDem_diagTRUE                                                14.237561
    log_edu                                                            0.109024
    edu_levelCollege or University degree                              0.225516
    edu_levelCSEs or equivalent                                        0.257416
    edu_levelNone of the above                                         0.487372
    edu_levelNVQ or HND or HNC or equivalent                           0.320160
    edu_levelO levels/GCSEs or equivalent                              0.069233
    edu_levelOther professional qualifications eg: nursing, teaching   0.304212
                                                                     Std. Error
    (Intercept)                                                        0.477290
    Population                                                         0.002323
    n_cancers                                                          0.108362
    townsend                                                           0.011949
    whr                                                                0.374575
    age                                                                0.004491
    peopleInHousehold                                                  0.021130
    ethnicity_simwhite                                                 0.096044
    PD_diagTrue                                                        0.269720
    AD_diagTrue                                                        0.325651
    vascDem_diagTRUE                                                   0.430448
    frontoDem_diagTRUE                                               196.967726
    log_edu                                                            0.033280
    edu_levelCollege or University degree                              0.242680
    edu_levelCSEs or equivalent                                        0.257745
    edu_levelNone of the above                                         0.234589
    edu_levelNVQ or HND or HNC or equivalent                           0.236533
    edu_levelO levels/GCSEs or equivalent                              0.238376
    edu_levelOther professional qualifications eg: nursing, teaching   0.228949
                                                                     z value
    (Intercept)                                                       -2.600
    Population                                                        -1.610
    n_cancers                                                         -1.884
    townsend                                                           2.253
    whr                                                                2.986
    age                                                               -6.904
    peopleInHousehold                                                  2.718
    ethnicity_simwhite                                                -3.306
    PD_diagTrue                                                        2.525
    AD_diagTrue                                                        4.699
    vascDem_diagTRUE                                                   3.423
    frontoDem_diagTRUE                                                 0.072
    log_edu                                                            3.276
    edu_levelCollege or University degree                              0.929
    edu_levelCSEs or equivalent                                        0.999
    edu_levelNone of the above                                         2.078
    edu_levelNVQ or HND or HNC or equivalent                           1.354
    edu_levelO levels/GCSEs or equivalent                              0.290
    edu_levelOther professional qualifications eg: nursing, teaching   1.329
                                                                     Pr(>|z|)    
    (Intercept)                                                      0.009318 ** 
    Population                                                       0.107372    
    n_cancers                                                        0.059610 .  
    townsend                                                         0.024238 *  
    whr                                                              0.002824 ** 
    age                                                              5.07e-12 ***
    peopleInHousehold                                                0.006577 ** 
    ethnicity_simwhite                                               0.000945 ***
    PD_diagTrue                                                      0.011560 *  
    AD_diagTrue                                                      2.61e-06 ***
    vascDem_diagTRUE                                                 0.000619 ***
    frontoDem_diagTRUE                                               0.942376    
    log_edu                                                          0.001053 ** 
    edu_levelCollege or University degree                            0.352748    
    edu_levelCSEs or equivalent                                      0.317930    
    edu_levelNone of the above                                       0.037750 *  
    edu_levelNVQ or HND or HNC or equivalent                         0.175879    
    edu_levelO levels/GCSEs or equivalent                            0.771481    
    edu_levelOther professional qualifications eg: nursing, teaching 0.183936    
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for binomial family taken to be 1)
    
        Null deviance: 6421.9  on 8799  degrees of freedom
    Residual deviance: 6223.3  on 8781  degrees of freedom
    AIC: 6261.3
    
    Number of Fisher Scoring iterations: 10
    



The reduced model is: <br>
glm(formula = result ~ Population + n_cancers + townsend + whr + 
    age + peopleInHousehold + ethnicity_sim + PD_diag + AD_diag + 
    vascDem_diag + frontoDem_diag + log_edu + edu_level, family = "binomial", 
    data = cur_dt_na)
<br>


```r
%%R

ukb_covid.b_red <-glm(formula = result ~ Population + n_cancers + townsend + whr + 
    age + peopleInHousehold + ethnicity_sim + PD_diag + AD_diag + 
    vascDem_diag + frontoDem_diag + log_edu + edu_level, family = "binomial", 
    data = cur_dt_na)
```

#### Comparison of the simplified model versus the full model


```r
%%R
anova(ukb_covid.b, ukb_covid.b_red, test="Chisq")
```


    Analysis of Deviance Table
    
    Model 1: result ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + AverageHouseholdIncome + 
        ethnicity_sim + highBP + PD_diag + AD_diag + vascDem_diag + 
        frontoDem_diag + log_edu + log_crp + blood_group + edu_level + 
        house_type
    Model 2: result ~ Population + n_cancers + townsend + whr + age + peopleInHousehold + 
        ethnicity_sim + PD_diag + AD_diag + vascDem_diag + frontoDem_diag + 
        log_edu + edu_level
      Resid. Df Resid. Dev  Df Deviance Pr(>Chi)
    1      8758     6205.7                      
    2      8781     6223.3 -23    -17.6   0.7788



This means that removing those variables does not significantly change the model.

#### Update the model with the full dataset. 


```r
%%R

ukb_covid.b_red <-glm(formula = result ~ Population + n_cancers + townsend + whr + 
    age + peopleInHousehold + ethnicity_sim + PD_diag + AD_diag + 
    vascDem_diag + frontoDem_diag + edu_level, family = "binomial", 
    data = cur_dt)

ukb_covid.b_red_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid.b_red), 
                           conf_int_low = summary(ukb_covid.b_red)$coefficients[,1] - 
                                          summary(ukb_covid.b_red)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid.b_red)$coefficients[,1] + 
                                          summary(ukb_covid.b_red)$coefficients[,2])),
                 p_value = summary(ukb_covid.b_red)$coefficients[,4]))

stargazer::stargazer(ukb_covid.b_red_odds, type ="html", 
                     single.row=TRUE, summary = FALSE, 
                     out = "fig_out/ukb_covid_b_red_odds.html")
write.csv(ukb_covid.b_red_odds,"data_output/ukb_covid_b_red_odds.csv", row.names=TRUE)

ukb_covid.b_red_odds$significance = "p-value > 0.05"
ukb_covid.b_red_odds$significance[ukb_covid.b_red_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid.b_red_odds$variables = row.names(ukb_covid.b_red_odds)

library(ggplot2)

ggplot(ukb_covid.b_red_odds, aes(x=reorder(variables, OR), y=OR, color=significance)) + 
    geom_point(fill="white", shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Infectivity odds ratios") + 
  xlab("Variables")
ggsave('fig_out/UKB_infectivityOR_plot_optimised.pdf', width = 8, height = 8)
ggsave('fig_out/UKB_infectivityOR_plot_optimised.png', width = 8, height = 8)
```


    
    <table style="text-align:center"><tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td>OR</td><td>conf_int_low</td><td>conf_int_high</td><td>p_value</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">(Intercept)</td><td>0.269</td><td>0.182</td><td>0.397</td><td>0.001</td></tr>
    <tr><td style="text-align:left">Population</td><td>0.998</td><td>0.996</td><td>0.999</td><td>0.150</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>0.810</td><td>0.742</td><td>0.885</td><td>0.017</td></tr>
    <tr><td style="text-align:left">townsend</td><td>1.038</td><td>1.029</td><td>1.047</td><td>0.00002</td></tr>
    <tr><td style="text-align:left">whr</td><td>3.088</td><td>2.273</td><td>4.195</td><td>0.0002</td></tr>
    <tr><td style="text-align:left">age</td><td>0.974</td><td>0.971</td><td>0.978</td><td>0</td></tr>
    <tr><td style="text-align:left">peopleInHousehold</td><td>1.054</td><td>1.037</td><td>1.072</td><td>0.002</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>0.721</td><td>0.668</td><td>0.778</td><td>0.00002</td></tr>
    <tr><td style="text-align:left">PD_diagTrue</td><td>1.712</td><td>1.371</td><td>2.138</td><td>0.016</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>4.153</td><td>3.225</td><td>5.349</td><td>0.00000</td></tr>
    <tr><td style="text-align:left">vascDem_diagTRUE</td><td>2.511</td><td>1.697</td><td>3.717</td><td>0.019</td></tr>
    <tr><td style="text-align:left">frontoDem_diagTRUE</td><td>3.090</td><td>1.161</td><td>8.221</td><td>0.249</td></tr>
    <tr><td style="text-align:left">edu_levelCollege or University degree</td><td>1.155</td><td>0.944</td><td>1.413</td><td>0.474</td></tr>
    <tr><td style="text-align:left">edu_levelCSEs or equivalent</td><td>1.413</td><td>1.143</td><td>1.747</td><td>0.103</td></tr>
    <tr><td style="text-align:left">edu_levelNone of the above</td><td>1.654</td><td>1.366</td><td>2.003</td><td>0.008</td></tr>
    <tr><td style="text-align:left">edu_levelNVQ or HND or HNC or equivalent</td><td>1.418</td><td>1.167</td><td>1.724</td><td>0.073</td></tr>
    <tr><td style="text-align:left">edu_levelO levels/GCSEs or equivalent</td><td>1.123</td><td>0.923</td><td>1.367</td><td>0.555</td></tr>
    <tr><td style="text-align:left">edu_levelOther professional qualifications eg: nursing, teaching</td><td>1.419</td><td>1.174</td><td>1.715</td><td>0.064</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr></table>



#### Create a model with only dementia - all diagnosis




```r
%%R
cur_dt_dem = subset(gen_dt, select = c(result, Population, n_cancers,townsend, smoking, whistling, diabetes,
                                 whr , sex , age ,peopleInHousehold,AverageHouseholdIncome,ethnicity_sim,highBP,
                          PD_diag, AD_diag , vascDem_diag,frontoDem_diag,allDem_diag,
#                           greyVol , whiteVol , wmh_norm,
                          edu_score, log_crp,blood_group,log_edu,
                          edu_level , house_type))

ukb_covid.b_red_dem <-glm(formula = result ~ Population + n_cancers + townsend + whr + 
    age + peopleInHousehold + ethnicity_sim + allDem_diag + edu_level+allDem_diag,
                          family = "binomial", 
    data = cur_dt_dem)

ukb_covid.b_red_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid.b_red_dem), 
                           conf_int_low = summary(ukb_covid.b_red_dem)$coefficients[,1] - 
                                          summary(ukb_covid.b_red_dem)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid.b_red_dem)$coefficients[,1] + 
                                          summary(ukb_covid.b_red_dem)$coefficients[,2])),
                 p_value = summary(ukb_covid.b_red_dem)$coefficients[,4]))

stargazer::stargazer(ukb_covid.b_red_dem, type ="html", 
                     single.row=TRUE, summary = FALSE, 
                     out = "fig_out/ukb_covid_b_dementia_model.html")
write.csv(ukb_covid.b_red_odds,"data_output/ukb_covid_b_red_odds_dementia.csv", row.names=TRUE)

ukb_covid.b_red_odds$significance = "p-value > 0.05"
ukb_covid.b_red_odds$significance[ukb_covid.b_red_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid.b_red_odds$variables = row.names(ukb_covid.b_red_odds)

library(ggplot2)

ggplot(ukb_covid.b_red_odds, aes(x=reorder(variables, OR), y=OR, color=significance)) + 
    geom_point(fill="white", shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Infectivity odds ratios") + 
  xlab("Variables")
# ggsave('fig_out/UKB_infectivityOR_plot_optimised_dementia.pdf', width = 8, height = 8)
ggsave('fig_out/UKB_infectivityOR_plot_optimised_dementia.png', width = 8, height = 8)

```


    
    <table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td><em>Dependent variable:</em></td></tr>
    <tr><td></td><td colspan="1" style="border-bottom: 1px solid black"></td></tr>
    <tr><td style="text-align:left"></td><td>result</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Population</td><td>-0.002 (0.002)</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>-0.204<sup>**</sup> (0.088)</td></tr>
    <tr><td style="text-align:left">townsend</td><td>0.036<sup>***</sup> (0.009)</td></tr>
    <tr><td style="text-align:left">whr</td><td>1.124<sup>***</sup> (0.306)</td></tr>
    <tr><td style="text-align:left">age</td><td>-0.026<sup>***</sup> (0.004)</td></tr>
    <tr><td style="text-align:left">peopleInHousehold</td><td>0.053<sup>***</sup> (0.017)</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>-0.327<sup>***</sup> (0.077)</td></tr>
    <tr><td style="text-align:left">allDem_diag</td><td>1.181<sup>***</sup> (0.174)</td></tr>
    <tr><td style="text-align:left">edu_levelCollege or University degree</td><td>0.150 (0.201)</td></tr>
    <tr><td style="text-align:left">edu_levelCSEs or equivalent</td><td>0.358<sup>*</sup> (0.212)</td></tr>
    <tr><td style="text-align:left">edu_levelNone of the above</td><td>0.515<sup>***</sup> (0.191)</td></tr>
    <tr><td style="text-align:left">edu_levelNVQ or HND or HNC or equivalent</td><td>0.357<sup>*</sup> (0.195)</td></tr>
    <tr><td style="text-align:left">edu_levelO levels/GCSEs or equivalent</td><td>0.121 (0.196)</td></tr>
    <tr><td style="text-align:left">edu_levelOther professional qualifications eg: nursing, teaching</td><td>0.356<sup>*</sup> (0.189)</td></tr>
    <tr><td style="text-align:left">Constant</td><td>-1.329<sup>***</sup> (0.388)</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>12,788</td></tr>
    <tr><td style="text-align:left">Log Likelihood</td><td>-4,618.582</td></tr>
    <tr><td style="text-align:left">Akaike Inf. Crit.</td><td>9,267.165</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
    </table>



#### Show the importance of dementia-related factors


```r
%%R
ukb_covid.b_red_noDem <-glm(formula = result ~ Population + n_cancers + townsend + whr + 
    age + peopleInHousehold + ethnicity_sim + edu_level, family = "binomial", data = cur_dt_na)

anova(ukb_covid.b, ukb_covid.b_red_noDem, test="Chisq")
```


    Analysis of Deviance Table
    
    Model 1: result ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + AverageHouseholdIncome + 
        ethnicity_sim + highBP + PD_diag + AD_diag + vascDem_diag + 
        frontoDem_diag + log_edu + log_crp + blood_group + edu_level + 
        house_type
    Model 2: result ~ Population + n_cancers + townsend + whr + age + peopleInHousehold + 
        ethnicity_sim + edu_level
      Resid. Df Resid. Dev  Df Deviance  Pr(>Chi)    
    1      8758     6205.7                           
    2      8786     6274.2 -28  -68.535 2.974e-05 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1



Taking out AD and PD significantly change the model. Dementia-related variables are therefore important for COVID-19-related cases. 

# Curation and analysis of COVID-19 death data

## Data curation and merging


```r
%%R

ukb_death = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/covid_refresh_16_8_2020/death_cause_17_8_2020.txt", sep = "\t")[c('eid', 'level', 'cause_icd10')]

ukb_dt = read.csv('data_output/ukb_covid_dt_out.csv', sep = ",")
     
ukb_dt_death = merge(ukb_dt, ukb_death, by = "eid", all.x = TRUE)

print(nrow((ukb_dt_death)))

```


    [1] 14769



Here, I look at the different causes of death. I realise that there are multiple causes and it seems unclear whether COVID-19 contributed to those deaths.


```r
%%R

ukb_death_code = read.csv("data/coding19.tsv", sep = "\t")[c('coding', 'meaning')]

# Here I create a for loop to loop through all ICD codings... because for some reason the built in function do not work

# subset the death code present in my dataset 

ukb_death_code_cleaned <- ukb_death_code[ukb_death_code$coding %in% na.omit(unique(ukb_dt_death$cause_icd10)), ]

ukb_dt_death$cause_icd10 <- ordered(ukb_dt_death$cause_icd10, levels = ukb_death_code_cleaned$coding, 
                                 labels = ukb_death_code_cleaned$meaning)
unique(na.omit(ukb_dt_death$cause_icd10))
```


      [1] I50.0 Congestive heart failure                                                                         
      [2] D46.9 Myelodysplastic syndrome, unspecified                                                            
      [3] C19 Malignant neoplasm of rectosigmoid junction                                                        
      [4] F01.9 Vascular dementia, unspecified                                                                   
      [5] U07.1 COVID-19 virus identified                                                                        
      [6] J12.9 Viral pneumonia, unspecified                                                                     
      [7] J44.9 Chronic obstructive pulmonary disease, unspecified                                               
      [8] J18.0 Bronchopneumonia, unspecified                                                                    
      [9] C34.9 Bronchus or lung, unspecified                                                                    
     [10] J18.9 Pneumonia, unspecified                                                                           
     [11] F03 Unspecified dementia                                                                               
     [12] N18.9 Chronic renal failure, unspecified                                                               
     [13] R68.8 Other specified general symptoms and signs                                                       
     [14] E11.9 Without complications                                                                            
     [15] E23.7 Disorder of pituitary gland, unspecified                                                         
     [16] Y83.9 Surgical procedure, unspecified                                                                  
     [17] R26.3 Immobility                                                                                       
     [18] T81.0 Haemorrhage and haematoma complicating a procedure, not elsewhere classified                     
     [19] W19 Unspecified fall                                                                                   
     [20] F10.1 Harmful use                                                                                      
     [21] S06.9 Intracranial injury, unspecified                                                                 
     [22] F41.9 Anxiety disorder, unspecified                                                                    
     [23] F32.9 Depressive episode, unspecified                                                                  
     [24] R54 Senility                                                                                           
     [25] K72.9 Hepatic failure, unspecified                                                                     
     [26] K74.3 Primary biliary cirrhosis                                                                        
     [27] K65.0 Acute peritonitis                                                                                
     [28] I10 Essential (primary) hypertension                                                                   
     [29] I26.9 Pulmonary embolism without mention of acute cor pulmonale                                        
     [30] J18.1 Lobar pneumonia, unspecified                                                                     
     [31] I21.9 Acute myocardial infarction, unspecified                                                         
     [32] J45.9 Asthma, unspecified                                                                              
     [33] I35.9 Aortic valve disorder, unspecified                                                               
     [34] E87.1 Hypo-osmolality and hyponatraemia                                                                
     [35] E10.9 Without complications                                                                            
     [36] I20.9 Angina pectoris, unspecified                                                                     
     [37] J96.9 Respiratory failure, unspecified                                                                 
     [38] I50.9 Heart failure, unspecified                                                                       
     [39] I70.9 Generalised and unspecified atherosclerosis                                                      
     [40] L03.9 Cellulitis, unspecified                                                                          
     [41] A41.9 Septicaemia, unspecified                                                                         
     [42] E66.9 Obesity, unspecified                                                                             
     [43] I89.0 Lymphoedema, not elsewhere classified                                                            
     [44] D61.9 Aplastic anaemia, unspecified                                                                    
     [45] C83.1 Small cleaved cell (diffuse)                                                                     
     [46] Y43.3 Other antineoplastic drugs                                                                       
     [47] D64.9 Anaemia, unspecified                                                                             
     [48] E78.0 Pure hypercholesterolaemia                                                                       
     [49] U07.2 COVID-19 virus not identified                                                                    
     [50] C92.0 Acute myeloid leukaemia                                                                          
     [51] I25.9 Chronic ischaemic heart disease, unspecified                                                     
     [52] I24.8 Other forms of acute ischaemic heart disease                                                     
     [53] J69.0 Pneumonitis due to food and vomit                                                                
     [54] G30.9 Alzheimer's disease, unspecified                                                                 
     [55] C50.9 Breast, unspecified                                                                              
     [56] J81 Pulmonary oedema                                                                                   
     [57] K21.9 Gastro-oesophageal reflux disease without oesophagitis                                           
     [58] G93.1 Anoxic brain damage, not elsewhere classified                                                    
     [59] J44.1 Chronic obstructive pulmonary disease with acute exacerbation, unspecified                       
     [60] I46.9 Cardiac arrest, unspecified                                                                      
     [61] I49.0 Ventricular fibrillation and flutter                                                             
     [62] C91.1 Chronic lymphocytic leukaemia                                                                    
     [63] C85.9 Non-Hodgkin's lymphoma, unspecified type                                                         
     [64] C80.9 Malignant neoplasm, unspecified                                                                  
     [65] N39.0 Urinary tract infection, site not specified                                                      
     [66] C16.9 Stomach, unspecified                                                                             
     [67] I80.2 Phlebitis and thrombophlebitis of other deep vessels of lower extremities                        
     [68] I62.9 Intracranial haemorrhage (nontraumatic), unspecified                                             
     [69] C79.3 Secondary malignant neoplasm of brain and cerebral meninges                                      
     [70] C43.9 Malignant melanoma of skin, unspecified                                                          
     [71] C79.9 Secondary malignant neoplasm, unspecified site                                                   
     [72] T14.9 Injury, unspecified                                                                              
     [73] I63.9 Cerebral infarction, unspecified                                                                 
     [74] C25.9 Pancreas, unspecified                                                                            
     [75] C92.1 Chronic myeloid leukaemia                                                                        
     [76] J10.1 Influenza with other respiratory manifestations, influenza virus identified                      
     [77] J10.0 Influenza with pneumonia, influenza virus identified                                             
     [78] C90.0 Multiple myeloma                                                                                 
     [79] C17.0 Duodenum                                                                                         
     [80] C15.9 Oesophagus, unspecified                                                                          
     [81] K55.0 Acute vascular disorders of intestine                                                            
     [82] E14.9 Without complications                                                                            
     [83] J80 Adult respiratory distress syndrome                                                                
     [84] C18.9 Colon, unspecified                                                                               
     [85] G30.1 Alzheimer's disease with late onset                                                              
     [86] N32.1 Vesicointestinal fistula                                                                         
     [87] K57.9 Diverticular disease of intestine, part unspecified, without perforation or abscess              
     [88] M34.9 Systemic sclerosis, unspecified                                                                  
     [89] I48.9 Atrial fibrillation and atrial flutter,  unspecified                                             
     [90] K51.9 Ulcerative colitis, unspecified                                                                  
     [91] J84.1 Other interstitial pulmonary diseases with fibrosis                                              
     [92] G47.3 Sleep apnoea                                                                                     
     [93] I51.6 Cardiovascular disease, unspecified                                                              
     [94] Z94.0 Kidney transplant status                                                                         
     [95] K86.8 Other specified diseases of pancreas                                                             
     [96] N18.5 Chronic kidney disease, stage 5                                                                  
     [97] Y43.4 Immunosuppressive agents                                                                         
     [98] B24 Unspecified human immunodeficiency virus [HIV] disease                                             
     [99] C67.9 Bladder, unspecified                                                                             
    [100] M35.3 Polymyalgia rheumatica                                                                           
    [101] J43.9 Emphysema, unspecified                                                                           
    [102] I48.0 Paroxysmal atrial fibrillation                                                                   
    [103] J15.9 Bacterial pneumonia, unspecified                                                                 
    [104] I74.9 Embolism and thrombosis of unspecified artery                                                    
    [105] M47.9 Spondylosis, unspecified                                                                         
    [106] K57.8 Diverticular disease of intestine, part unspecified, with perforation and abscess                
    [107] K74.6 Other and unspecified cirrhosis of liver                                                         
    [108] G20 Parkinson's disease                                                                                
    [109] S72.0 Fracture of neck of femur                                                                        
    [110] Z51.9 Medical care, unspecified                                                                        
    [111] E78.5 Hyperlipidaemia, unspecified                                                                     
    [112] C79.5 Secondary malignant neoplasm of bone and bone marrow                                             
    [113] C61 Malignant neoplasm of prostate                                                                     
    [114] E03.9 Hypothyroidism, unspecified                                                                      
    [115] G31.8 Other specified degenerative diseases of nervous system                                          
    [116] K70.3 Alcoholic cirrhosis of liver                                                                     
    [117] B18.2 Chronic viral hepatitis C                                                                        
    [118] I50.1 Left ventricular failure                                                                         
    [119] E66.8 Other obesity                                                                                    
    [120] K70.0 Alcoholic fatty liver                                                                            
    [121] I11.0 Hypertensive heart disease with (congestive) heart failure                                       
    [122] I11.9 Hypertensive heart disease without (congestive) heart failure                                    
    [123] C80.0 Malignant neoplasm, primary site unknown, so stated                                              
    [124] I64 Stroke, not specified as haemorrhage or infarction                                                 
    [125] K92.2 Gastro-intestinal haemorrhage, unspecified                                                       
    [126] C64 Malignant neoplasm of kidney, except renal pelvis                                                  
    [127] F10.9 Unspecified mental and behavioural disorder                                                      
    [128] I85.9 Oesophageal varices without bleeding                                                             
    [129] K76.0 Fatty (change of) liver, not elsewhere classified                                                
    [130] C71.9 Brain, unspecified                                                                               
    [131] I51.8 Other ill-defined heart diseases                                                                 
    [132] C85.1 B-cell lymphoma, unspecified                                                                     
    [133] C90.1 Plasma cell leukaemia                                                                            
    [134] M35.0 Sicca syndrome [Sjogren]                                                                         
    [135] I45.9 Conduction disorder, unspecified                                                                 
    [136] C26.0 Intestinal tract, part unspecified                                                               
    [137] I27.2 Other secondary pulmonary hypertension                                                           
    [138] M34.8 Other forms of systemic sclerosis                                                                
    [139] I67.9 Cerebrovascular disease, unspecified                                                             
    [140] J22 Unspecified acute lower respiratory infection                                                      
    [141] J44.0 Chronic obstructive pulmonary disease with acute lower respiratory infection                     
    [142] J12.8 Other viral pneumonia                                                                            
    [143] D41.4 Bladder                                                                                          
    [144] Z95.1 Presence of aortocoronary bypass graft                                                           
    [145] Z95.0 Presence of cardiac pacemaker                                                                    
    [146] I49.9 Cardiac arrhythmia, unspecified                                                                  
    [147] I25.1 Atherosclerotic heart disease                                                                    
    [148] J15.4 Pneumonia due to other streptococci                                                              
    [149] I69.4 Sequelae of stroke, not specified as haemorrhage or infarction                                   
    [150] Q85.0 Neurofibromatosis (nonmalignant)                                                                 
    [151] N17.9 Acute renal failure, unspecified                                                                 
    [152] I38 Endocarditis, valve unspecified                                                                    
    [153] I82.9 Embolism and thrombosis of unspecified vein                                                      
    [154] M10.9 Gout, unspecified                                                                                
    [155] N04.9 Unspecified                                                                                      
    [156] N28.9 Disorder of kidney and ureter, unspecified                                                       
    [157] E85.4 Organ-limited amyloidosis                                                                        
    [158] N03.9 Unspecified                                                                                      
    [159] D50.9 Iron deficiency anaemia, unspecified                                                             
    [160] J84.9 Interstitial pulmonary disease, unspecified                                                      
    [161] K56.6 Other and unspecified intestinal obstruction                                                     
    [162] G04.9 Encephalitis, myelitis and encephalomyelitis, unspecified                                        
    [163] D76.1 Haemophagocytic lymphohistiocytosis                                                              
    [164] K83.0 Cholangitis                                                                                      
    [165] C22.9 Liver, unspecified                                                                               
    [166] S32.5 Fracture of pubis                                                                                
    [167] X59.0 Home                                                                                             
    [168] C78.7 Secondary malignant neoplasm of liver                                                            
    [169] C73 Malignant neoplasm of thyroid gland                                                                
    [170] K28.9 Unspecified as acute or chronic, without haemorrhage or perforation                              
    [171] K28.4 Chronic or unspecified with haemorrhage                                                          
    [172] K80.5 Calculus of bile duct without cholangitis or cholecystitis                                       
    [173] K27.9 Unspecified as acute or chronic, without haemorrhage or perforation                              
    [174] K27.4 Chronic or unspecified with haemorrhage                                                          
    [175] K66.0 Peritoneal adhesions                                                                             
    [176] K43.3 Parastomal hernia with obstruction,  without gangrene                                            
    [177] K63.9 Disease of intestine, unspecified                                                                
    [178] I06.0 Rheumatic aortic stenosis                                                                        
    [179] I05.1 Rheumatic mitral insufficiency                                                                   
    [180] I21.4 Acute subendocardial myocardial infarction                                                       
    [181] K43.5 Parastomal hernia without obstruction or gangrene                                                
    [182] Z90.4 Acquired absence of other parts of digestive tract                                               
    [183] F50.0 Anorexia nervosa                                                                                 
    [184] N61 Inflammatory disorders of breast                                                                   
    [185] I42.9 Cardiomyopathy, unspecified                                                                      
    [186] K80.2 Calculus of gallbladder without cholecystitis                                                    
    [187] K82.2 Perforation of gallbladder                                                                       
    [188] K76.7 Hepatorenal syndrome                                                                             
    [189] C22.1 Intrahepatic bile duct carcinoma                                                                 
    [190] K57.2 Diverticular disease of large intestine with perforation and abscess                             
    [191] K65.9 Peritonitis, unspecified                                                                         
    [192] R57.0 Cardiogenic shock                                                                                
    [193] M85.8 Other specified disorders of bone density and structure                                          
    [194] I72.2 Aneurysm of renal artery                                                                         
    [195] I71.9 Aortic aneurysm of unspecified site, without mention of rupture                                  
    [196] I77.2 Rupture of artery                                                                                
    [197] K70.4 Alcoholic hepatic failure                                                                        
    [198] I73.9 Peripheral vascular disease, unspecified                                                         
    [199] D69.0 Allergic purpura                                                                                 
    [200] A49.8 Other bacterial infections of unspecified site                                                   
    [201] I25.5 Ischaemic cardiomyopathy                                                                         
    [202] M86.9 Osteomyelitis, unspecified                                                                       
    [203] K81.9 Cholecystitis, unspecified                                                                       
    [204] I33.0 Acute and subacute infective endocarditis                                                        
    [205] N19 Unspecified renal failure                                                                          
    [206] C18.7 Sigmoid colon                                                                                    
    [207] K56.7 Ileus, unspecified                                                                               
    [208] K56.2 Volvulus                                                                                         
    [209] Z51.8 Other specified medical care                                                                     
    [210] J93.9 Pneumothorax, unspecified                                                                        
    [211] K55.9 Vascular disorder of intestine, unspecified                                                      
    [212] C49.9 Connective and soft tissue, unspecified                                                          
    [213] M13.9 Arthritis, unspecified                                                                           
    [214] D51.0 Vitamin B12 deficiency anaemia due to intrinsic factor deficiency                                
    [215] J98.9 Respiratory disorder, unspecified                                                                
    [216] K59.0 Constipation                                                                                     
    [217] K70.9 Alcoholic liver disease, unspecified                                                             
    [218] C20 Malignant neoplasm of rectum                                                                       
    [219] I00 Rheumatic fever without mention of heart involvement                                               
    [220] Z95.2 Presence of prosthetic heart valve                                                               
    [221] C18.1 Appendix                                                                                         
    [222] Q27.9 Congenital malformation of peripheral vascular system, unspecified                               
    [223] C18.0 Caecum                                                                                           
    [224] K80.4 Calculus of bile duct with cholecystitis                                                         
    [225] S22.4 Multiple fractures of ribs                                                                       
    [226] N40 Hyperplasia of prostate                                                                            
    [227] M19.9 Arthrosis, unspecified                                                                           
    [228] G91.2 Normal-pressure hydrocephalus                                                                    
    [229] I70.1 Atherosclerosis of renal artery                                                                  
    [230] C78.0 Secondary malignant neoplasm of lung                                                             
    [231] G45.9 Transient cerebral ischaemic attack, unspecified                                                 
    [232] M81.9 Osteoporosis, unspecified                                                                        
    [233] M06.9 Rheumatoid arthritis, unspecified                                                                
    [234] Y83.5 Amputation of limb(s)                                                                            
    [235] T87.4 Infection of amputation stump                                                                    
    [236] M60.0 Infective myositis                                                                               
    [237] I35.0 Aortic (valve) stenosis                                                                          
    [238] E27.4 Other and unspecified adrenocortical insufficiency                                               
    [239] K75.0 Abscess of liver                                                                                 
    [240] C93.1 Chronic monocytic leukaemia                                                                      
    [241] I71.1 Thoracic aortic aneurysm, ruptured                                                               
    [242] I31.2 Haemopericardium, not elsewhere classified                                                       
    [243] M41.9 Scoliosis, unspecified                                                                           
    [244] Z96.6 Presence of orthopaedic joint implants                                                           
    [245] A49.0 Staphylococcal infection, unspecified                                                            
    [246] R99 Other ill-defined and unspecified causes of mortality                                              
    [247] R00.1 Bradycardia, unspecified                                                                         
    [248] M25.9 Joint disorder, unspecified                                                                      
    [249] C37 Malignant neoplasm of thymus                                                                       
    [250] J86.0 Pyothorax with fistula                                                                           
    [251] C34.1 Upper lobe, bronchus or lung                                                                     
    [252] C56 Malignant neoplasm of ovary                                                                        
    [253] F31.9 Bipolar affective disorder, unspecified                                                          
    [254] C24.1 Ampulla of Vater                                                                                 
    [255] E63.9 Nutritional deficiency, unspecified                                                              
    [256] W80 Inhalation and ingestion of other objects causing obstruction of respiratory tract                 
    [257] T17.9 Foreign body in respiratory tract, part unspecified                                              
    [258] C16.0 Cardia                                                                                           
    [259] C81.9 Hodgkin's disease, unspecified                                                                   
    [260] D35.2 Pituitary gland                                                                                  
    [261] G12.2 Motor neuron disease                                                                             
    [262] I12.0 Hypertensive renal disease with renal failure                                                    
    [263] Y84.1 Kidney dialysis                                                                                  
    [264] B37.8 Candidiasis of other sites                                                                       
    [265] T85.7 Infection and inflammatory reaction due to other internal prosthetic devices, implants and grafts
    [266] D70 Agranulocytosis                                                                                    
    [267] I60.8 Other subarachnoid haemorrhage                                                                   
    [268] I61.9 Intracerebral haemorrhage, unspecified                                                           
    [269] N28.0 Ischaemia and infarction of kidney                                                               
    [270] C22.0 Liver cell carcinoma                                                                             
    [271] D37.2 Small intestine                                                                                  
    [272] D59.3 Haemolytic-uraemic syndrome                                                                      
    [273] J98.0 Diseases of bronchus, not elsewhere classified                                                   
    [274] R04.0 Epistaxis                                                                                        
    [275] C78.6 Secondary malignant neoplasm of retroperitoneum and peritoneum                                   
    [276] C82.9 Follicular non-Hodgkin's lymphoma, unspecified                                                   
    [277] C02.9 Tongue, unspecified                                                                              
    [278] C55 Malignant neoplasm of uterus, part unspecified                                                     
    [279] D80.1 Nonfamilial hypogammaglobulinaemia                                                               
    [280] J47 Bronchiectasis                                                                                     
    [281] G70.0 Myasthenia gravis                                                                                
    [282] Z90.8 Acquired absence of other organs                                                                 
    [283] G35 Multiple sclerosis                                                                                 
    [284] B34.2 Coronavirus infection, unspecified                                                               
    [285] M32.9 Systemic lupus erythematosus, unspecified                                                        
    [286] I27.9 Pulmonary heart disease, unspecified                                                             
    [287] C83.0 Small cell (diffuse)                                                                             
    [288] Y83.4 Other reconstructive surgery                                                                     
    [289] Y44.2 Anticoagulants                                                                                   
    [290] K56.5 Intestinal adhesions [bands] with obstruction                                                    
    [291] G61.0 Guillain-Barre syndrome                                                                          
    [292] I69.1 Sequelae of intracerebral haemorrhage                                                            
    [293] G83.9 Paralytic syndrome, unspecified                                                                  
    [294] Y83.1 Surgical operation with implant of artificial internal device                                    
    [295] I05.9 Mitral valve disease, unspecified                                                                
    [296] M00.9 Pyogenic arthritis, unspecified                                                                  
    [297] C23 Malignant neoplasm of gallbladder                                                                  
    [298] I42.0 Dilated cardiomyopathy                                                                           
    [299] J60 Coalworker's pneumoconiosis                                                                        
    [300] F20.9 Schizophrenia, unspecified                                                                       
    [301] A49.1 Streptococcal infection, unspecified                                                             
    [302] I71.0 Dissection of aorta [any part]                                                                   
    [303] A04.7 Enterocolitis due to Clostridium difficile                                                       
    [304] I24.9 Acute ischaemic heart disease, unspecified                                                       
    [305] I13.2 Hypertensive heart and renal disease with both (congestive) heart failure and renal failure      
    [306] I13.9 Hypertensive heart and renal disease, unspecified                                                
    [307] C45.9 Mesothelioma, unspecified                                                                        
    [308] R09.2 Respiratory arrest                                                                               
    [309] M35.9 Systemic involvement of connective tissue, unspecified                                           
    [310] C76.2 Abdomen                                                                                          
    [311] R57.2 Septic shock                                                                                     
    [312] C84.4 Peripheral T-cell lymphoma                                                                       
    [313] C54.1 Endometrium                                                                                      
    [314] I42.1 Obstructive hypertrophic cardiomyopathy                                                          
    [315] D32.0 Cerebral meninges                                                                                
    [316] J18.2 Hypostatic pneumonia, unspecified                                                                
    [317] G52.1 Disorders of glossopharyngeal nerve                                                              
    [318] C44.9 Malignant neoplasm of skin, unspecified                                                          
    [319] I44.2 Atrioventricular block, complete                                                                 
    [320] R02 Gangrene, not elsewhere classified                                                                 
    [321] C79.8 Secondary malignant neoplasm of other specified sites                                            
    [322] C10.9 Oropharynx, unspecified                                                                          
    [323] Z93.4 Other artificial openings of gastro-intestinal tract status                                      
    [324] Y83.6 Removal of other organ (partial) (total)                                                         
    [325] R13 Dysphagia                                                                                          
    [326] V89.2 Person injured in unspecified motor-vehicle accident, traffic                                    
    [327] K92.0 Haematemesis                                                                                     
    [328] G10 Huntington's disease                                                                               
    [329] G61.8 Other inflammatory polyneuropathies                                                              
    [330] A41.5 Septicaemia due to other Gram-negative organisms                                                 
    [331] Y84.8 Other medical procedures                                                                         
    [332] T81.4 Infection following a procedure, not elsewhere classified                                        
    [333] X59.9 Unspecified place                                                                                
    [334] S14.1 Other and unspecified injuries of cervical spinal cord                                           
    [335] C95.9 Leukaemia, unspecified                                                                           
    [336] I60.9 Subarachnoid haemorrhage, unspecified                                                            
    [337] L40.9 Psoriasis, unspecified                                                                           
    [338] G23.2 Striatonigral degeneration                                                                       
    [339] R19.8 Other specified symptoms and signs involving the digestive system and abdomen                    
    [340] K66.9 Disorder of peritoneum, unspecified                                                              
    [341] D40.0 Prostate                                                                                         
    [342] C12 Malignant neoplasm of pyriform sinus                                                               
    [343] E11.5 With peripheral circulatory complications                                                        
    [344] K83.1 Obstruction of bile duct                                                                         
    [345] L40.5 Arthropathic psoriasis                                                                           
    [346] G40.9 Epilepsy, unspecified                                                                            
    [347] G30.0 Alzheimer's disease with early onset                                                             
    [348] M45 Ankylosing spondylitis                                                                             
    [349] G60.0 Hereditary motor and sensory neuropathy                                                          
    [350] K85.9 Acute pancreatitis, unspecified                                                                  
    [351] I40.9 Acute myocarditis, unspecified                                                                   
    [352] E11.7 With multiple complications                                                                      
    [353] K63.1 Perforation of intestine (nontraumatic)                                                          
    [354] E14.2 Withrenal complications                                                                          
    [355] T86.4 Liver transplant failure and rejection                                                           
    [356] Y83.0 Surgical operation with transplant of whole organ                                                
    [357] E87.2 Acidosis                                                                                         
    [358] R41.8 Other and unspecified symptoms and signs involving cognitive functions and awareness             
    [359] C83.3 Large cell (diffuse)                                                                             
    [360] F43.9 Reaction to severe stress, unspecified                                                           
    [361] C75.5 Aortic body and other paraganglia                                                                
    [362] M80.9 Unspecified osteoporosis with pathological fracture                                              
    [363] K86.1 Other chronic pancreatitis                                                                       
    363 Levels: A04.7 Enterocolitis due to Clostridium difficile < ...



Here, I will consider that a death is related to COVID-19 if the participant tested positive for COVID-19 and died. <br> This approach is anostic to the reported cause of death of the participant, and considers that COVID-19 can cause deat via different means.


```r
%%R

ukb_death = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/covid_refresh_16_8_2020/death_cause_17_8_2020.txt", sep = "\t")[c('eid', 'level', 'cause_icd10')]

ukb_dt = read.csv('data_output/ukb_covid_dt_out.csv', sep = ",")
     
ukb_dt_death = merge(ukb_dt, ukb_death, by = "eid")

ukb_deaths_eids = unique(ukb_dt_death$eid)


ukb_dt$death = ukb_dt$eid %in% ukb_deaths_eids

print(sum(ukb_dt$death))

write.csv(ukb_dt,"data_output/ukb_covid_addDeath.csv", row.names=FALSE)
```


    [1] 647



### Calculate the percentage of deaths related to COVID-19 & odds ratio

67% of deaths were associated with COVID-19


```r
%%R

ukb_dt = read.csv("data_output/ukb_covid_addDeath.csv")
ukb_dt_died = subset(ukb_dt, death == TRUE)

print(sum(ukb_dt_died$result)/sum(ukb_dt$death))
```


    [1] 0.4621329



### Mortality odds while acccounting for other factors including COVID-19

#### Scale all numeric data and curate ethnicity value


```r
%%R

ukb_dt$peopleInHousehold = scale(ukb_dt$peopleInHousehold)
ukb_dt$age = scale(ukb_dt$age)
ukb_dt$whr = scale(ukb_dt$whr)
ukb_dt$pm25_val = scale(ukb_dt$pm25_val)
ukb_dt$no2_val = scale(ukb_dt$no2_val)
ukb_dt$nox_val = scale(ukb_dt$nox_val)
ukb_dt$so2_val = scale(ukb_dt$so2_val)
ukb_dt$pm10_val = scale(ukb_dt$pm10_val)
ukb_dt$o3_val = scale(ukb_dt$o3_val)
```

#### General model on death, all causes


```r
%%R

ukb_dt_subset = subset(ukb_dt, select = c(result,death, Population, n_cancers,townsend, smoking, whistling, diabetes,
                                 whr , sex , age ,peopleInHousehold,AverageHouseholdIncome,ethnicity_sim,highBP,
                          PD_diag, AD_diag , vascDem_diag,frontoDem_diag,
#                           greyVol , whiteVol , wmh_norm,
                          edu_score, crp_norm,log_crp,log_edu,
                          edu_level , house_type))

ukb_dt_subset_na = na.omit(ukb_dt_subset)
summary(ukb_covid_death.b <-glm(data = ukb_dt_subset_na, death ~ Population + n_cancers +townsend + smoking + whistling + diabetes+
                                 whr + sex + age +peopleInHousehold+AverageHouseholdIncome+ethnicity_sim+highBP+
                          PD_diag+ AD_diag + vascDem_diag+frontoDem_diag+
#                           greyVol + whiteVol + wmh_norm+
                          edu_score+ log_crp + log_edu + house_type, family = 'binomial'))

stargazer::stargazer(ukb_covid_death.b, type ="html", 
                     single.row=TRUE,
                     out = "fig_out/ukb_covid_death_coefficients_fullmodel.html")

```


    
    <table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td><em>Dependent variable:</em></td></tr>
    <tr><td></td><td colspan="1" style="border-bottom: 1px solid black"></td></tr>
    <tr><td style="text-align:left"></td><td>death</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Population</td><td>0.002 (0.002)</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>-0.140 (0.147)</td></tr>
    <tr><td style="text-align:left">townsend</td><td>0.005 (0.022)</td></tr>
    <tr><td style="text-align:left">smokingNever</td><td>-0.518<sup>***</sup> (0.157)</td></tr>
    <tr><td style="text-align:left">smokingPrevious</td><td>-0.505<sup>***</sup> (0.154)</td></tr>
    <tr><td style="text-align:left">whistling</td><td>0.153 (0.116)</td></tr>
    <tr><td style="text-align:left">diabetes</td><td>0.255<sup>*</sup> (0.153)</td></tr>
    <tr><td style="text-align:left">whr</td><td>0.410 (0.797)</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>0.368<sup>***</sup> (0.142)</td></tr>
    <tr><td style="text-align:left">age</td><td>0.075<sup>***</sup> (0.009)</td></tr>
    <tr><td style="text-align:left">peopleInHousehold</td><td>0.036 (0.035)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome18,000 to 30,999</td><td>0.451 (1,674.884)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome31,000 to 51,999</td><td>0.534 (1,674.884)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome52,000 to 100,000</td><td>0.818 (1,674.884)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeGreater than 100,000</td><td>-0.106 (1,674.884)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeLess than 18,000</td><td>0.742 (1,674.884)</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>-0.114 (0.169)</td></tr>
    <tr><td style="text-align:left">highBPFalse</td><td>0.089 (0.399)</td></tr>
    <tr><td style="text-align:left">highBPTrue</td><td>0.349 (0.398)</td></tr>
    <tr><td style="text-align:left">PD_diagTrue</td><td>-0.136 (0.469)</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>1.472<sup>***</sup> (0.366)</td></tr>
    <tr><td style="text-align:left">vascDem_diag</td><td>0.476 (0.582)</td></tr>
    <tr><td style="text-align:left">frontoDem_diag</td><td>18.008 (1,455.398)</td></tr>
    <tr><td style="text-align:left">edu_score</td><td>0.003 (0.005)</td></tr>
    <tr><td style="text-align:left">log_crp</td><td>0.155<sup>***</sup> (0.051)</td></tr>
    <tr><td style="text-align:left">log_edu</td><td>-0.035 (0.075)</td></tr>
    <tr><td style="text-align:left">house_typeA house or bungalow</td><td>-0.090 (0.182)</td></tr>
    <tr><td style="text-align:left">house_typeCare home</td><td>-13.033 (1,455.398)</td></tr>
    <tr><td style="text-align:left">house_typeMobile or temporary structure (i.e. caravan)</td><td>-12.774 (392.309)</td></tr>
    <tr><td style="text-align:left">house_typeNone of the above</td><td>-12.505 (295.194)</td></tr>
    <tr><td style="text-align:left">house_typeSheltered accommodation</td><td>-12.846 (828.888)</td></tr>
    <tr><td style="text-align:left">Constant</td><td>-8.740 (1,674.884)</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>8,897</td></tr>
    <tr><td style="text-align:left">Log Likelihood</td><td>-1,545.468</td></tr>
    <tr><td style="text-align:left">Akaike Inf. Crit.</td><td>3,154.936</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
    </table>



Calculate odds ratios


```r
%%R
ukb_covid_death.b_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid_death.b), 
                           conf_int_low = summary(ukb_covid_death.b)$coefficients[,1] - 
                                          summary(ukb_covid_death.b)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid_death.b)$coefficients[,1] + 
                                          summary(ukb_covid_death.b)$coefficients[,2])),
                 p_value = summary(ukb_covid_death.b)$coefficients[,4]))

stargazer::stargazer(ukb_covid_death.b_odds, type ="html", 
                     single.row=TRUE, summary = FALSE, 
                     out = "fig_out/ukb_covid_death_odds_fullmodel.html")
write.csv(ukb_covid_death.b_odds,"data_output/ukb_covid_allDeath_odds.csv", row.names=TRUE)

ukb_covid_death.b_odds$significance = "p-value > 0.05"
ukb_covid_death.b_odds$significance[ukb_covid_death.b_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid_death.b_odds$variables = row.names(ukb_covid_death.b_odds)

library(ggplot2)
ggplot(ukb_covid_death.b_odds[which(ukb_covid_death.b_odds$significance == "p-value < 0.05"),], aes(x=reorder(variables, OR), y=OR)) + 
    geom_point(shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Mortality odds ratios") + 
  xlab("Variables")
ggsave('fig_out/UKB_deathOR_plot_sig_only.pdf', width = 6, height = 4)
ggsave('fig_out/UKB_deathOR_plot_sig_only.png', width = 6, height = 4)
```


    
    <table style="text-align:center"><tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td>OR</td><td>conf_int_low</td><td>conf_int_high</td><td>p_value</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">(Intercept)</td><td>0.0002</td><td>0</td><td>Inf</td><td>0.996</td></tr>
    <tr><td style="text-align:left">Population</td><td>1.002</td><td>1.000</td><td>1.004</td><td>0.305</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>0.870</td><td>0.751</td><td>1.007</td><td>0.342</td></tr>
    <tr><td style="text-align:left">townsend</td><td>1.005</td><td>0.983</td><td>1.027</td><td>0.837</td></tr>
    <tr><td style="text-align:left">smokingNever</td><td>0.596</td><td>0.509</td><td>0.697</td><td>0.001</td></tr>
    <tr><td style="text-align:left">smokingPrevious</td><td>0.604</td><td>0.518</td><td>0.704</td><td>0.001</td></tr>
    <tr><td style="text-align:left">whistlingTRUE</td><td>1.165</td><td>1.038</td><td>1.308</td><td>0.186</td></tr>
    <tr><td style="text-align:left">diabetesTRUE</td><td>1.291</td><td>1.108</td><td>1.504</td><td>0.095</td></tr>
    <tr><td style="text-align:left">whr</td><td>1.507</td><td>0.679</td><td>3.344</td><td>0.607</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>1.445</td><td>1.254</td><td>1.666</td><td>0.010</td></tr>
    <tr><td style="text-align:left">age</td><td>1.078</td><td>1.068</td><td>1.087</td><td>0</td></tr>
    <tr><td style="text-align:left">peopleInHousehold</td><td>1.036</td><td>1.001</td><td>1.073</td><td>0.303</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome18,000 to 30,999</td><td>1.570</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome31,000 to 51,999</td><td>1.705</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome52,000 to 100,000</td><td>2.266</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeGreater than 100,000</td><td>0.900</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeLess than 18,000</td><td>2.101</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>0.892</td><td>0.753</td><td>1.057</td><td>0.500</td></tr>
    <tr><td style="text-align:left">highBPFalse</td><td>1.094</td><td>0.734</td><td>1.630</td><td>0.823</td></tr>
    <tr><td style="text-align:left">highBPTrue</td><td>1.417</td><td>0.953</td><td>2.109</td><td>0.380</td></tr>
    <tr><td style="text-align:left">PD_diagTrue</td><td>0.873</td><td>0.546</td><td>1.395</td><td>0.771</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>4.360</td><td>3.022</td><td>6.289</td><td>0.0001</td></tr>
    <tr><td style="text-align:left">vascDem_diagTRUE</td><td>1.610</td><td>0.900</td><td>2.882</td><td>0.413</td></tr>
    <tr><td style="text-align:left">frontoDem_diagTRUE</td><td>66,155,689.000</td><td>0</td><td>Inf</td><td>0.990</td></tr>
    <tr><td style="text-align:left">edu_score</td><td>1.003</td><td>0.997</td><td>1.009</td><td>0.588</td></tr>
    <tr><td style="text-align:left">log_crp</td><td>1.167</td><td>1.109</td><td>1.228</td><td>0.002</td></tr>
    <tr><td style="text-align:left">log_edu</td><td>0.965</td><td>0.895</td><td>1.041</td><td>0.641</td></tr>
    <tr><td style="text-align:left">house_typeA house or bungalow</td><td>0.914</td><td>0.763</td><td>1.096</td><td>0.622</td></tr>
    <tr><td style="text-align:left">house_typeCare home</td><td>0.00000</td><td>0</td><td>Inf</td><td>0.993</td></tr>
    <tr><td style="text-align:left">house_typeMobile or temporary structure (i.e. caravan)</td><td>0.00000</td><td>0</td><td>6</td><td>0.974</td></tr>
    <tr><td style="text-align:left">house_typeNone of the above</td><td>0.00000</td><td>0</td><td>5</td><td>0.966</td></tr>
    <tr><td style="text-align:left">house_typeSheltered accommodation</td><td>0.00000</td><td>0</td><td>Inf</td><td>0.988</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr></table>



This shows that COVID-19 is a significant predictor of the risk of dying.

### Main contributors of COVID-19-related death


```r
%%R
ukb_dt = read.csv("data_output/ukb_covid_addDeath.csv")

ukb_dt$covid_death = ifelse(ukb_dt$result == 1 & ukb_dt$death == TRUE, 
1, 0)
print(sum(ukb_dt$covid_death))

write.csv(ukb_dt,"data_output/ukb_covid_addDeath.csv",row.names=FALSE)
```


    [1] 299



There are here 299 participants that died and tested positive for COVID-19

#### Dementia-only model - COVID death


```r
%%R
ukb_dt = read.csv("data_output/ukb_covid_addDeath.csv")

ukb_dt_dementia = subset(ukb_dt, select = c(result,covid_death,death, Population, n_cancers,townsend, smoking, whistling, diabetes,
                                 whr , sex , age ,peopleInHousehold,AverageHouseholdIncome,ethnicity_sim,highBP,
                          PD_diag, AD_diag , vascDem_diag,frontoDem_diag,allDem_diag,
#                           greyVol , whiteVol , wmh_norm,
                          edu_score, crp_norm,log_crp,log_edu,
                          edu_level , house_type))
```


      result covid_death death Population n_cancers    townsend  smoking whistling
    1      0           0 FALSE   3.541627         0 -5.87237000 Previous     FALSE
    2      0           0 FALSE   2.932501         0  2.91871000 Previous     FALSE
    3      0           0 FALSE  10.947637         0  3.43245000    Never     FALSE
    4      0           0  TRUE   6.636715         0 -2.30303000 Previous     FALSE
    5      0           0 FALSE  11.849605         0 -0.00099431 Previous     FALSE
    6      0           0 FALSE   3.617117         0 -3.92656000    Never     FALSE
      diabetes       whr    sex age peopleInHousehold AverageHouseholdIncome
    1    FALSE 0.8514851   Male  70                 2                   <NA>
    2    FALSE 0.9090909 Female  72                 2       18,000 to 30,999
    3    FALSE 0.8942308 Female  53                 1                   <NA>
    4    FALSE 0.9626168   Male  81                 2       18,000 to 30,999
    5    FALSE 0.9207921 Female  73                 1       18,000 to 30,999
    6    FALSE 1.0256410   Male  75                NA                   <NA>
      ethnicity_sim highBP PD_diag AD_diag vascDem_diag frontoDem_diag allDem_diag
    1         white  False   False   False        FALSE          FALSE       FALSE
    2         white   True   False   False        FALSE          FALSE       FALSE
    3      minority  False   False   False        FALSE          FALSE       FALSE
    4         white   True   False   False        FALSE          FALSE       FALSE
    5         white  False   False   False        FALSE          FALSE       FALSE
    6         white   True   False   False        FALSE          FALSE       FALSE
      edu_score    crp_norm   log_crp   log_edu
    1      3.57 0.127777778 -2.057463 1.2725656
    2     23.23 0.060053619 -2.812517 3.1454445
    3     17.37 0.003726548 -5.592273 2.8547446
    4     25.85 0.022968705 -3.773623 3.2523106
    5      2.96 0.036278039 -3.316543 1.0851893
    6      1.45 0.030045435 -3.505045 0.3715636
                                                    edu_level
    1                         NVQ or HND or HNC or equivalent
    2                                       None of the above
    3 Other professional qualifications eg: nursing, teaching
    4                            College or University degree
    5 Other professional qualifications eg: nursing, teaching
    6                                                    <NA>
                           house_type
    1             A house or bungalow
    2             A house or bungalow
    3 A flat, maisonette or apartment
    4             A house or bungalow
    5             A house or bungalow
    6             A house or bungalow




```r
%%R

# I took out AverageHouseholdIncome and house_type because they are not significant, and their confidence intervals
# are so large that I cannot plot it

summary(ukb_covid_and_death.b <-glm(data = ukb_dt_dementia, covid_death ~ Population + n_cancers +townsend + smoking + whistling + diabetes+
                                 whr + sex + age +peopleInHousehold+ethnicity_sim+highBP+
                          allDem_diag+
#                           greyVol + whiteVol + wmh_norm+
                          log_crp, family = 'binomial'))

stargazer::stargazer(ukb_covid_and_death.b, type ="html", 
                     single.row=TRUE,
                     out = "fig_out/ukb_deathAndCOVID_coefficients_dementia_model.html")

ukb_covid_death_covid.b_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid_and_death.b), 
                           conf_int_low = summary(ukb_covid_and_death.b)$coefficients[,1] - 
                                          summary(ukb_covid_and_death.b)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid_and_death.b)$coefficients[,1] + 
                                          summary(ukb_covid_and_death.b)$coefficients[,2])),
                 p_value = summary(ukb_covid_and_death.b)$coefficients[,4]))

write.csv(ukb_covid_death_covid.b_odds,"data_output/ukb_deathAndCOVID_odds_dementia.csv", row.names=TRUE)

ukb_covid_death_covid.b_odds$significance = "p-value > 0.05"
ukb_covid_death_covid.b_odds$significance[ukb_covid_death_covid.b_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid_death_covid.b_odds$variables = row.names(ukb_covid_death_covid.b_odds)

library(ggplot2)
ggplot(ukb_covid_death_covid.b_odds, 
       aes(x=reorder(variables, OR), y=OR, color = significance)) + 
    geom_point(shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Mortality odds ratios, death and COVID-19") + 
  xlab("")
ggsave('fig_out/UKB_deathAndCOVID_ORplot_dementiaModel.png', width = 6, height = 4)
```


    
    <table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td><em>Dependent variable:</em></td></tr>
    <tr><td></td><td colspan="1" style="border-bottom: 1px solid black"></td></tr>
    <tr><td style="text-align:left"></td><td>covid_death</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Population</td><td>0.002 (0.003)</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>-0.564<sup>**</sup> (0.236)</td></tr>
    <tr><td style="text-align:left">townsend</td><td>0.072<sup>***</sup> (0.020)</td></tr>
    <tr><td style="text-align:left">smokingNever</td><td>-0.158 (0.221)</td></tr>
    <tr><td style="text-align:left">smokingPrevious</td><td>-0.085 (0.214)</td></tr>
    <tr><td style="text-align:left">whistling</td><td>0.058 (0.152)</td></tr>
    <tr><td style="text-align:left">diabetes</td><td>0.067 (0.197)</td></tr>
    <tr><td style="text-align:left">whr</td><td>1.306 (1.023)</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>0.367<sup>**</sup> (0.184)</td></tr>
    <tr><td style="text-align:left">age</td><td>0.090<sup>***</sup> (0.012)</td></tr>
    <tr><td style="text-align:left">peopleInHousehold</td><td>0.012 (0.050)</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>-0.333<sup>*</sup> (0.197)</td></tr>
    <tr><td style="text-align:left">highBPFalse</td><td>-0.260 (0.435)</td></tr>
    <tr><td style="text-align:left">highBPTrue</td><td>0.045 (0.431)</td></tr>
    <tr><td style="text-align:left">allDem_diag</td><td>1.465<sup>***</sup> (0.260)</td></tr>
    <tr><td style="text-align:left">log_crp</td><td>0.102 (0.067)</td></tr>
    <tr><td style="text-align:left">Constant</td><td>-10.892<sup>***</sup> (1.350)</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>10,821</td></tr>
    <tr><td style="text-align:left">Log Likelihood</td><td>-1,035.227</td></tr>
    <tr><td style="text-align:left">Akaike Inf. Crit.</td><td>2,104.454</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
    </table>



#### Dementia subtypes


```r
%%R
ukb_dt_subset = subset(ukb_dt, select = c(result,covid_death,death, Population, n_cancers,townsend, smoking, whistling, diabetes,
                                 whr , sex , age ,peopleInHousehold,AverageHouseholdIncome,ethnicity_sim,highBP,
                          PD_diag, AD_diag , vascDem_diag,frontoDem_diag,
#                           greyVol , whiteVol , wmh_norm,
                          edu_score, crp_norm,log_crp,log_edu,
                          edu_level , house_type,
                          pm25_val,no2_val,nox_val,so2_val,pm10_val,o3_val))
ukb_dt_subset_na = na.omit(ukb_dt_subset)
summary(ukb_covid_and_death.b <-glm(data = ukb_dt_subset_na, covid_death ~ Population + n_cancers +townsend + smoking + whistling + diabetes+
                                 whr + sex + age +peopleInHousehold+AverageHouseholdIncome+ethnicity_sim+highBP+
                          PD_diag+ AD_diag + vascDem_diag+frontoDem_diag+
#                           greyVol + whiteVol + wmh_norm+
                          edu_score+ log_crp+ house_type, family = 'binomial'))

stargazer::stargazer(ukb_covid_and_death.b, type ="html", 
                     single.row=TRUE,
                     out = "fig_out/ukb_covid_deathAndCOVID_coefficients_fullmodel.html")

ukb_covid_death_covid.b_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid_and_death.b), 
                           conf_int_low = summary(ukb_covid_and_death.b)$coefficients[,1] - 
                                          summary(ukb_covid_and_death.b)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid_and_death.b)$coefficients[,1] + 
                                          summary(ukb_covid_and_death.b)$coefficients[,2])),
                 p_value = summary(ukb_covid_and_death.b)$coefficients[,4]))

stargazer::stargazer(ukb_covid_death_covid.b_odds, type ="html", 
                     single.row=TRUE, summary = FALSE, 
                     out = "fig_out/ukb_covid_deathAndCOVID_odds_fullmodel.html")
write.csv(ukb_covid_death.b_odds,"data_output/ukb_covid_deathAndCOVID_odds.csv", row.names=TRUE)

ukb_covid_death_covid.b_odds$significance = "p-value > 0.05"
ukb_covid_death_covid.b_odds$significance[ukb_covid_death_covid.b_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid_death_covid.b_odds$variables = row.names(ukb_covid_death_covid.b_odds)

library(ggplot2)
ggplot(ukb_covid_death_covid.b_odds[which(ukb_covid_death_covid.b_odds$significance == "p-value < 0.05"),], 
       aes(x=reorder(variables, OR), y=OR)) + 
    geom_point(shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Mortality odds ratios, death and COVID-19") + 
  xlab("Variables")
ggsave('fig_out/UKB_deathAndCOVID_ORplot_sig_only.pdf', width = 6, height = 4)
ggsave('fig_out/UKB_deathAndCOVID_ORplot_sig_only.png', width = 6, height = 4)
```


    
    <table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td><em>Dependent variable:</em></td></tr>
    <tr><td></td><td colspan="1" style="border-bottom: 1px solid black"></td></tr>
    <tr><td style="text-align:left"></td><td>covid_death</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Population</td><td>0.002 (0.003)</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>-0.381 (0.244)</td></tr>
    <tr><td style="text-align:left">townsend</td><td>0.068<sup>**</sup> (0.031)</td></tr>
    <tr><td style="text-align:left">smokingNever</td><td>-0.124 (0.254)</td></tr>
    <tr><td style="text-align:left">smokingPrevious</td><td>0.009 (0.244)</td></tr>
    <tr><td style="text-align:left">whistling</td><td>0.086 (0.172)</td></tr>
    <tr><td style="text-align:left">diabetes</td><td>0.197 (0.214)</td></tr>
    <tr><td style="text-align:left">whr</td><td>1.853 (1.170)</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>0.399<sup>*</sup> (0.213)</td></tr>
    <tr><td style="text-align:left">age</td><td>0.088<sup>***</sup> (0.014)</td></tr>
    <tr><td style="text-align:left">peopleInHousehold</td><td>0.033 (0.054)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome18,000 to 30,999</td><td>0.282 (2,758.201)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome31,000 to 51,999</td><td>0.770 (2,758.201)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome52,000 to 100,000</td><td>0.705 (2,758.201)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeGreater than 100,000</td><td>-0.307 (2,758.201)</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeLess than 18,000</td><td>0.754 (2,758.201)</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>-0.357 (0.224)</td></tr>
    <tr><td style="text-align:left">highBPFalse</td><td>-0.169 (0.527)</td></tr>
    <tr><td style="text-align:left">highBPTrue</td><td>0.062 (0.525)</td></tr>
    <tr><td style="text-align:left">PD_diagTrue</td><td>0.409 (0.537)</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>1.738<sup>***</sup> (0.442)</td></tr>
    <tr><td style="text-align:left">vascDem_diag</td><td>0.832 (0.670)</td></tr>
    <tr><td style="text-align:left">frontoDem_diag</td><td>19.358 (2,399.545)</td></tr>
    <tr><td style="text-align:left">edu_score</td><td>-0.003 (0.005)</td></tr>
    <tr><td style="text-align:left">log_crp</td><td>0.102 (0.077)</td></tr>
    <tr><td style="text-align:left">house_typeA house or bungalow</td><td>-0.369 (0.240)</td></tr>
    <tr><td style="text-align:left">house_typeCare home</td><td>-14.119 (2,399.545)</td></tr>
    <tr><td style="text-align:left">house_typeMobile or temporary structure (i.e. caravan)</td><td>-12.958 (643.231)</td></tr>
    <tr><td style="text-align:left">house_typeNone of the above</td><td>-12.831 (486.051)</td></tr>
    <tr><td style="text-align:left">house_typeSheltered accommodation</td><td>-13.383 (1,360.094)</td></tr>
    <tr><td style="text-align:left">Constant</td><td>-11.670 (2,758.201)</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>8,897</td></tr>
    <tr><td style="text-align:left">Log Likelihood</td><td>-804.906</td></tr>
    <tr><td style="text-align:left">Akaike Inf. Crit.</td><td>1,671.811</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
    </table>
    
    <table style="text-align:center"><tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td>OR</td><td>conf_int_low</td><td>conf_int_high</td><td>p_value</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">(Intercept)</td><td>0.00001</td><td>0</td><td>Inf</td><td>0.997</td></tr>
    <tr><td style="text-align:left">Population</td><td>1.002</td><td>0.999</td><td>1.005</td><td>0.561</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>0.683</td><td>0.535</td><td>0.872</td><td>0.118</td></tr>
    <tr><td style="text-align:left">townsend</td><td>1.070</td><td>1.038</td><td>1.104</td><td>0.027</td></tr>
    <tr><td style="text-align:left">smokingNever</td><td>0.883</td><td>0.685</td><td>1.139</td><td>0.626</td></tr>
    <tr><td style="text-align:left">smokingPrevious</td><td>1.009</td><td>0.791</td><td>1.289</td><td>0.969</td></tr>
    <tr><td style="text-align:left">whistlingTRUE</td><td>1.089</td><td>0.918</td><td>1.293</td><td>0.618</td></tr>
    <tr><td style="text-align:left">diabetesTRUE</td><td>1.218</td><td>0.984</td><td>1.508</td><td>0.356</td></tr>
    <tr><td style="text-align:left">whr</td><td>6.379</td><td>1.980</td><td>20.556</td><td>0.113</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>1.491</td><td>1.205</td><td>1.844</td><td>0.060</td></tr>
    <tr><td style="text-align:left">age</td><td>1.092</td><td>1.077</td><td>1.107</td><td>0</td></tr>
    <tr><td style="text-align:left">peopleInHousehold</td><td>1.034</td><td>0.980</td><td>1.091</td><td>0.536</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome18,000 to 30,999</td><td>1.326</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome31,000 to 51,999</td><td>2.160</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncome52,000 to 100,000</td><td>2.024</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeGreater than 100,000</td><td>0.736</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">AverageHouseholdIncomeLess than 18,000</td><td>2.126</td><td>0</td><td>Inf</td><td>1.000</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>0.700</td><td>0.559</td><td>0.876</td><td>0.111</td></tr>
    <tr><td style="text-align:left">highBPFalse</td><td>0.845</td><td>0.499</td><td>1.432</td><td>0.749</td></tr>
    <tr><td style="text-align:left">highBPTrue</td><td>1.064</td><td>0.630</td><td>1.798</td><td>0.905</td></tr>
    <tr><td style="text-align:left">PD_diagTrue</td><td>1.505</td><td>0.880</td><td>2.575</td><td>0.446</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>5.684</td><td>3.655</td><td>8.840</td><td>0.0001</td></tr>
    <tr><td style="text-align:left">vascDem_diagTRUE</td><td>2.299</td><td>1.176</td><td>4.492</td><td>0.214</td></tr>
    <tr><td style="text-align:left">frontoDem_diagTRUE</td><td>255,357,389.000</td><td>0</td><td>Inf</td><td>0.994</td></tr>
    <tr><td style="text-align:left">edu_score</td><td>0.997</td><td>0.992</td><td>1.002</td><td>0.591</td></tr>
    <tr><td style="text-align:left">log_crp</td><td>1.107</td><td>1.026</td><td>1.196</td><td>0.182</td></tr>
    <tr><td style="text-align:left">house_typeA house or bungalow</td><td>0.691</td><td>0.544</td><td>0.879</td><td>0.124</td></tr>
    <tr><td style="text-align:left">house_typeCare home</td><td>0.00000</td><td>0</td><td>Inf</td><td>0.995</td></tr>
    <tr><td style="text-align:left">house_typeMobile or temporary structure (i.e. caravan)</td><td>0.00000</td><td>0</td><td>5</td><td>0.984</td></tr>
    <tr><td style="text-align:left">house_typeNone of the above</td><td>0.00000</td><td>0</td><td>3</td><td>0.979</td></tr>
    <tr><td style="text-align:left">house_typeSheltered accommodation</td><td>0.00000</td><td>0</td><td>Inf</td><td>0.992</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr></table>



#### Simplification of the COVID death model 

Note that stepAIC must not have any NAs


```r
%%R
library(MASS)

ukb_covid_and_death.b_step <- stepAIC(ukb_covid_and_death.b, direction = "both")
summary(ukb_covid_and_death.b_step)
```


    Start:  AIC=1671.81
    covid_death ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + AverageHouseholdIncome + 
        ethnicity_sim + highBP + PD_diag + AD_diag + vascDem_diag + 
        frontoDem_diag + edu_score + log_crp + log_edu + +house_type
    
                             Df Deviance    AIC
    - house_type              5   1611.7 1665.7
    - smoking                 2   1608.5 1668.5
    - highBP                  2   1610.0 1670.0
    - whistling               1   1608.0 1670.0
    - Population              1   1608.1 1670.1
    - peopleInHousehold       1   1608.1 1670.1
    - edu_score               1   1608.3 1670.3
    - PD_diag                 1   1608.4 1670.4
    - diabetes                1   1608.7 1670.7
    - AverageHouseholdIncome  5   1617.1 1671.1
    - vascDem_diag            1   1609.1 1671.1
    - log_crp                 1   1609.7 1671.7
    - log_edu                 1   1609.8 1671.8
    <none>                        1607.8 1671.8
    - ethnicity_sim           1   1610.3 1672.3
    - whr                     1   1610.4 1672.4
    - n_cancers               1   1610.7 1672.7
    - sex                     1   1611.3 1673.3
    - townsend                1   1612.8 1674.8
    - frontoDem_diag          1   1613.2 1675.2
    - AD_diag                 1   1619.2 1681.2
    - age                     1   1653.4 1715.4
    
    Step:  AIC=1665.65
    covid_death ~ Population + n_cancers + townsend + smoking + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + AverageHouseholdIncome + 
        ethnicity_sim + highBP + PD_diag + AD_diag + vascDem_diag + 
        frontoDem_diag + edu_score + log_crp + log_edu
    
                             Df Deviance    AIC
    - smoking                 2   1612.4 1662.4
    - peopleInHousehold       1   1611.8 1663.8
    - highBP                  2   1613.8 1663.8
    - edu_score               1   1611.8 1663.8
    - whistling               1   1611.9 1663.9
    - Population              1   1612.0 1664.0
    - PD_diag                 1   1612.2 1664.2
    - diabetes                1   1612.6 1664.6
    - vascDem_diag            1   1613.0 1665.0
    - log_crp                 1   1613.5 1665.5
    <none>                        1611.7 1665.7
    - AverageHouseholdIncome  5   1621.7 1665.7
    - log_edu                 1   1613.8 1665.8
    - ethnicity_sim           1   1614.3 1666.3
    - whr                     1   1614.4 1666.4
    - n_cancers               1   1614.5 1666.5
    - sex                     1   1615.4 1667.4
    - frontoDem_diag          1   1616.9 1668.9
    + house_type              5   1607.8 1671.8
    - townsend                1   1622.6 1674.6
    - AD_diag                 1   1622.8 1674.8
    - age                     1   1655.7 1707.7
    
    Step:  AIC=1662.38
    covid_death ~ Population + n_cancers + townsend + whistling + 
        diabetes + whr + sex + age + peopleInHousehold + AverageHouseholdIncome + 
        ethnicity_sim + highBP + PD_diag + AD_diag + vascDem_diag + 
        frontoDem_diag + edu_score + log_crp + log_edu
    
                             Df Deviance    AIC
    - peopleInHousehold       1   1612.5 1660.5
    - highBP                  2   1614.5 1660.5
    - edu_score               1   1612.6 1660.6
    - Population              1   1612.7 1660.7
    - whistling               1   1612.8 1660.8
    - PD_diag                 1   1612.8 1660.8
    - diabetes                1   1613.3 1661.3
    - vascDem_diag            1   1613.7 1661.7
    <none>                        1612.4 1662.4
    - log_crp                 1   1614.4 1662.4
    - AverageHouseholdIncome  5   1622.5 1662.5
    - log_edu                 1   1614.5 1662.5
    - ethnicity_sim           1   1615.0 1663.0
    - n_cancers               1   1615.2 1663.2
    - whr                     1   1615.4 1663.4
    - sex                     1   1616.5 1664.5
    + smoking                 2   1611.7 1665.7
    - frontoDem_diag          1   1617.8 1665.8
    + house_type              5   1608.5 1668.5
    - AD_diag                 1   1623.5 1671.5
    - townsend                1   1623.6 1671.6
    - age                     1   1658.2 1706.2
    
    Step:  AIC=1660.5
    covid_death ~ Population + n_cancers + townsend + whistling + 
        diabetes + whr + sex + age + AverageHouseholdIncome + ethnicity_sim + 
        highBP + PD_diag + AD_diag + vascDem_diag + frontoDem_diag + 
        edu_score + log_crp + log_edu
    
                             Df Deviance    AIC
    - highBP                  2   1614.6 1658.6
    - edu_score               1   1612.7 1658.7
    - Population              1   1612.8 1658.8
    - whistling               1   1612.9 1658.9
    - PD_diag                 1   1613.0 1659.0
    - diabetes                1   1613.4 1659.4
    - vascDem_diag            1   1613.8 1659.8
    - AverageHouseholdIncome  5   1622.5 1660.5
    <none>                        1612.5 1660.5
    - log_crp                 1   1614.5 1660.5
    - log_edu                 1   1614.7 1660.7
    - ethnicity_sim           1   1615.2 1661.2
    - n_cancers               1   1615.3 1661.3
    - whr                     1   1615.5 1661.5
    + peopleInHousehold       1   1612.4 1662.4
    - sex                     1   1616.6 1662.6
    + smoking                 2   1611.8 1663.8
    - frontoDem_diag          1   1617.9 1663.9
    + house_type              5   1608.8 1666.8
    - townsend                1   1623.6 1669.6
    - AD_diag                 1   1623.6 1669.6
    - age                     1   1659.2 1705.2
    
    Step:  AIC=1658.59
    covid_death ~ Population + n_cancers + townsend + whistling + 
        diabetes + whr + sex + age + AverageHouseholdIncome + ethnicity_sim + 
        PD_diag + AD_diag + vascDem_diag + frontoDem_diag + edu_score + 
        log_crp + log_edu
    
                             Df Deviance    AIC
    - edu_score               1   1614.8 1656.8
    - Population              1   1614.9 1656.9
    - whistling               1   1615.0 1657.0
    - PD_diag                 1   1615.0 1657.0
    - diabetes                1   1615.4 1657.4
    - vascDem_diag            1   1616.0 1658.0
    <none>                        1614.6 1658.6
    - log_edu                 1   1616.7 1658.7
    - log_crp                 1   1616.7 1658.7
    - AverageHouseholdIncome  5   1624.9 1658.9
    - ethnicity_sim           1   1617.4 1659.4
    - n_cancers               1   1617.4 1659.4
    - whr                     1   1617.8 1659.8
    + highBP                  2   1612.5 1660.5
    + peopleInHousehold       1   1614.5 1660.5
    - sex                     1   1618.9 1660.9
    + smoking                 2   1613.9 1661.9
    - frontoDem_diag          1   1620.2 1662.2
    + house_type              5   1610.9 1664.9
    - townsend                1   1625.2 1667.2
    - AD_diag                 1   1625.9 1667.9
    - age                     1   1664.5 1706.5
    
    Step:  AIC=1656.83
    covid_death ~ Population + n_cancers + townsend + whistling + 
        diabetes + whr + sex + age + AverageHouseholdIncome + ethnicity_sim + 
        PD_diag + AD_diag + vascDem_diag + frontoDem_diag + log_crp + 
        log_edu
    
                             Df Deviance    AIC
    - Population              1   1615.1 1655.1
    - whistling               1   1615.2 1655.2
    - PD_diag                 1   1615.3 1655.3
    - diabetes                1   1615.6 1655.6
    - vascDem_diag            1   1616.2 1656.2
    <none>                        1614.8 1656.8
    - log_crp                 1   1617.0 1657.0
    - AverageHouseholdIncome  5   1625.2 1657.2
    - ethnicity_sim           1   1617.5 1657.5
    - log_edu                 1   1617.6 1657.6
    - n_cancers               1   1617.6 1657.6
    - whr                     1   1618.0 1658.0
    + edu_score               1   1614.6 1658.6
    + highBP                  2   1612.7 1658.7
    + peopleInHousehold       1   1614.7 1658.7
    - sex                     1   1619.2 1659.2
    + smoking                 2   1614.1 1660.1
    - frontoDem_diag          1   1620.4 1660.4
    + house_type              5   1611.4 1663.4
    - AD_diag                 1   1626.0 1666.0
    - townsend                1   1627.6 1667.6
    - age                     1   1664.7 1704.7
    
    Step:  AIC=1655.13
    covid_death ~ n_cancers + townsend + whistling + diabetes + whr + 
        sex + age + AverageHouseholdIncome + ethnicity_sim + PD_diag + 
        AD_diag + vascDem_diag + frontoDem_diag + log_crp + log_edu
    
                             Df Deviance    AIC
    - whistling               1   1615.5 1653.5
    - PD_diag                 1   1615.6 1653.6
    - diabetes                1   1615.9 1653.9
    - vascDem_diag            1   1616.5 1654.5
    <none>                        1615.1 1655.1
    - log_crp                 1   1617.3 1655.3
    - AverageHouseholdIncome  5   1625.5 1655.5
    - ethnicity_sim           1   1617.8 1655.8
    - n_cancers               1   1618.0 1656.0
    - log_edu                 1   1618.0 1656.0
    - whr                     1   1618.3 1656.3
    + Population              1   1614.8 1656.8
    + edu_score               1   1614.9 1656.9
    + highBP                  2   1613.0 1657.0
    + peopleInHousehold       1   1615.0 1657.0
    - sex                     1   1619.5 1657.5
    + smoking                 2   1614.4 1658.4
    - frontoDem_diag          1   1620.7 1658.7
    + house_type              5   1611.7 1661.7
    - AD_diag                 1   1626.5 1664.5
    - townsend                1   1628.5 1666.5
    - age                     1   1664.8 1702.8
    
    Step:  AIC=1653.51
    covid_death ~ n_cancers + townsend + diabetes + whr + sex + age + 
        AverageHouseholdIncome + ethnicity_sim + PD_diag + AD_diag + 
        vascDem_diag + frontoDem_diag + log_crp + log_edu
    
                             Df Deviance    AIC
    - PD_diag                 1   1616.0 1652.0
    - diabetes                1   1616.3 1652.3
    - vascDem_diag            1   1617.0 1653.0
    <none>                        1615.5 1653.5
    - log_crp                 1   1617.9 1653.9
    - AverageHouseholdIncome  5   1625.9 1653.9
    - ethnicity_sim           1   1618.1 1654.1
    - n_cancers               1   1618.3 1654.3
    - log_edu                 1   1618.3 1654.3
    - whr                     1   1619.0 1655.0
    + whistling               1   1615.1 1655.1
    + Population              1   1615.2 1655.2
    + edu_score               1   1615.3 1655.3
    + peopleInHousehold       1   1615.4 1655.4
    + highBP                  2   1613.5 1655.5
    - sex                     1   1619.8 1655.8
    + smoking                 2   1614.7 1656.7
    - frontoDem_diag          1   1621.0 1657.0
    + house_type              5   1612.0 1660.0
    - AD_diag                 1   1626.8 1662.8
    - townsend                1   1629.1 1665.1
    - age                     1   1665.2 1701.2
    
    Step:  AIC=1651.97
    covid_death ~ n_cancers + townsend + diabetes + whr + sex + age + 
        AverageHouseholdIncome + ethnicity_sim + AD_diag + vascDem_diag + 
        frontoDem_diag + log_crp + log_edu
    
                             Df Deviance    AIC
    - diabetes                1   1616.8 1650.8
    - vascDem_diag            1   1617.5 1651.5
    <none>                        1616.0 1652.0
    - log_crp                 1   1618.3 1652.3
    - AverageHouseholdIncome  5   1626.5 1652.5
    - ethnicity_sim           1   1618.6 1652.6
    - n_cancers               1   1618.7 1652.7
    - log_edu                 1   1618.8 1652.8
    + PD_diag                 1   1615.5 1653.5
    - whr                     1   1619.5 1653.5
    + whistling               1   1615.6 1653.6
    + Population              1   1615.7 1653.7
    + edu_score               1   1615.8 1653.8
    + peopleInHousehold       1   1615.9 1653.9
    + highBP                  2   1613.9 1653.9
    - sex                     1   1620.3 1654.3
    + smoking                 2   1615.2 1655.2
    - frontoDem_diag          1   1621.5 1655.5
    + house_type              5   1612.5 1658.5
    - AD_diag                 1   1627.1 1661.1
    - townsend                1   1629.7 1663.7
    - age                     1   1666.1 1700.1
    
    Step:  AIC=1650.77
    covid_death ~ n_cancers + townsend + whr + sex + age + AverageHouseholdIncome + 
        ethnicity_sim + AD_diag + vascDem_diag + frontoDem_diag + 
        log_crp + log_edu
    
                             Df Deviance    AIC
    - vascDem_diag            1   1618.5 1650.5
    <none>                        1616.8 1650.8
    - log_crp                 1   1619.1 1651.1
    - AverageHouseholdIncome  5   1627.4 1651.4
    - log_edu                 1   1619.5 1651.5
    - n_cancers               1   1619.5 1651.5
    - ethnicity_sim           1   1619.5 1651.5
    + diabetes                1   1616.0 1652.0
    + PD_diag                 1   1616.3 1652.3
    + whistling               1   1616.4 1652.4
    + Population              1   1616.5 1652.5
    + edu_score               1   1616.6 1652.6
    + peopleInHousehold       1   1616.7 1652.7
    - sex                     1   1620.8 1652.8
    + highBP                  2   1614.8 1652.8
    - whr                     1   1621.2 1653.2
    + smoking                 2   1616.0 1654.0
    - frontoDem_diag          1   1622.2 1654.2
    + house_type              5   1613.3 1657.3
    - AD_diag                 1   1628.2 1660.2
    - townsend                1   1630.8 1662.8
    - age                     1   1667.7 1699.7
    
    Step:  AIC=1650.47
    covid_death ~ n_cancers + townsend + whr + sex + age + AverageHouseholdIncome + 
        ethnicity_sim + AD_diag + frontoDem_diag + log_crp + log_edu
    
                             Df Deviance    AIC
    <none>                        1618.5 1650.5
    + vascDem_diag            1   1616.8 1650.8
    - log_crp                 1   1620.8 1650.8
    - AverageHouseholdIncome  5   1629.0 1651.0
    - n_cancers               1   1621.2 1651.2
    - ethnicity_sim           1   1621.3 1651.3
    - log_edu                 1   1621.4 1651.4
    + diabetes                1   1617.5 1651.5
    + PD_diag                 1   1617.9 1651.9
    + whistling               1   1618.1 1652.1
    + Population              1   1618.1 1652.1
    + edu_score               1   1618.3 1652.3
    + peopleInHousehold       1   1618.3 1652.3
    - sex                     1   1622.4 1652.4
    + highBP                  2   1616.5 1652.5
    - whr                     1   1623.2 1653.2
    + smoking                 2   1617.8 1653.8
    - frontoDem_diag          1   1623.9 1653.9
    + house_type              5   1614.9 1656.9
    - AD_diag                 1   1630.8 1660.8
    - townsend                1   1632.5 1662.5
    - age                     1   1670.1 1700.1
    
    Call:
    glm(formula = covid_death ~ n_cancers + townsend + whr + sex + 
        age + AverageHouseholdIncome + ethnicity_sim + AD_diag + 
        frontoDem_diag + log_crp + log_edu, family = "binomial", 
        data = ukb_dt_subset_na)
    
    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -0.9767  -0.2305  -0.1643  -0.1084   3.4518  
    
    Coefficients:
                                                Estimate Std. Error z value
    (Intercept)                                -22.30594  535.41339  -0.042
    n_cancers                                   -0.37601    0.24317  -1.546
    townsend                                     0.09685    0.02535   3.821
    whr                                          2.44369    1.12285   2.176
    sexMale                                      0.40982    0.20927   1.958
    age                                          0.08768    0.01331   6.589
    AverageHouseholdIncome18,000 to 30,999      10.41858  535.41129   0.019
    AverageHouseholdIncome31,000 to 51,999      10.87387  535.41130   0.020
    AverageHouseholdIncome52,000 to 100,000     10.76709  535.41134   0.020
    AverageHouseholdIncomeGreater than 100,000   9.67883  535.41177   0.018
    AverageHouseholdIncomeLess than 18,000      10.94271  535.41126   0.020
    ethnicity_simwhite                          -0.38248    0.21965  -1.741
    AD_diagTrue                                  1.82048    0.43361   4.198
    frontoDem_diagTRUE                          16.25771  535.41123   0.030
    log_crp                                      0.11620    0.07562   1.537
    log_edu                                     -0.12269    0.07075  -1.734
                                               Pr(>|z|)    
    (Intercept)                                0.966769    
    n_cancers                                  0.122043    
    townsend                                   0.000133 ***
    whr                                        0.029531 *  
    sexMale                                    0.050193 .  
    age                                        4.44e-11 ***
    AverageHouseholdIncome18,000 to 30,999     0.984475    
    AverageHouseholdIncome31,000 to 51,999     0.983797    
    AverageHouseholdIncome52,000 to 100,000    0.983956    
    AverageHouseholdIncomeGreater than 100,000 0.985577    
    AverageHouseholdIncomeLess than 18,000     0.983694    
    ethnicity_simwhite                         0.081636 .  
    AD_diagTrue                                2.69e-05 ***
    frontoDem_diagTRUE                         0.975776    
    log_crp                                    0.124371    
    log_edu                                    0.082875 .  
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for binomial family taken to be 1)
    
        Null deviance: 1783.7  on 8896  degrees of freedom
    Residual deviance: 1618.5  on 8881  degrees of freedom
    AIC: 1650.5
    
    Number of Fisher Scoring iterations: 12
    



The simplified model is:

glm(formula = covid_death ~ n_cancers + townsend + whr + sex + 
    age + AverageHouseholdIncome + ethnicity_sim + AD_diag + 
    frontoDem_diag + log_crp + log_edu + so2_val, family = "binomial", 
    data = ukb_dt_subset_na)

#### Simplified COVID death model


```r
%%R

ukb_covid_death_covid_red.b = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + AD_diag + frontoDem_diag + 
    log_crp,
    family = "binomial", data = ukb_dt)

stargazer::stargazer(ukb_covid_death_covid_red.b, type ="html", 
                     single.row=TRUE,
                     out = "fig_out/ukb_covid_deathAndCOVID_coefficients_simplifiedModel.html")

ukb_covid_death_covid_red.b_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid_death_covid_red.b), 
                           conf_int_low = summary(ukb_covid_death_covid_red.b)$coefficients[,1] - 
                                          summary(ukb_covid_death_covid_red.b)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid_death_covid_red.b)$coefficients[,1] + 
                                          summary(ukb_covid_death_covid_red.b)$coefficients[,2])),
                 p_value = summary(ukb_covid_death_covid_red.b)$coefficients[,4]))

stargazer::stargazer(ukb_covid_death_covid_red.b_odds, type ="html", 
                     single.row=TRUE, summary = FALSE, 
                     out = "fig_out/ukb_covid_deathAndCOVID_odds_simplifiedModel.html")
write.csv(ukb_covid_death_covid_red.b_odds,"data_output/ukb_covid_deathAndCOVID_odds_simplifiedModel.csv", row.names=TRUE)

ukb_covid_death_covid_red.b_odds$significance = "p-value > 0.05"
ukb_covid_death_covid_red.b_odds$significance[ukb_covid_death_covid_red.b_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid_death_covid_red.b_odds$variables = row.names(ukb_covid_death_covid_red.b_odds)

library(ggplot2)
ggplot(ukb_covid_death_covid_red.b_odds, 
       aes(x=reorder(variables, OR), y=OR, color = significance)) + 
    geom_point(shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Mortality odds ratios, death and COVID-19, simplified model") + 
  xlab("")
ggsave('fig_out/UKB_deathAndCOVID_ORplot_simplifiedModel.png', width = 6, height = 4)
```


    
    <table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td><em>Dependent variable:</em></td></tr>
    <tr><td></td><td colspan="1" style="border-bottom: 1px solid black"></td></tr>
    <tr><td style="text-align:left"></td><td>covid_death</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">n_cancers</td><td>-0.449<sup>**</sup> (0.218)</td></tr>
    <tr><td style="text-align:left">townsend</td><td>0.070<sup>***</sup> (0.019)</td></tr>
    <tr><td style="text-align:left">diabetes</td><td>0.087 (0.190)</td></tr>
    <tr><td style="text-align:left">whr</td><td>1.763<sup>*</sup> (0.983)</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>0.329<sup>*</sup> (0.176)</td></tr>
    <tr><td style="text-align:left">age</td><td>0.089<sup>***</sup> (0.011)</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>-0.274 (0.190)</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>1.428<sup>***</sup> (0.372)</td></tr>
    <tr><td style="text-align:left">frontoDem_diag</td><td>2.795<sup>**</sup> (1.100)</td></tr>
    <tr><td style="text-align:left">log_crp</td><td>0.103 (0.065)</td></tr>
    <tr><td style="text-align:left">Constant</td><td>-11.384<sup>***</sup> (1.189)</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>11,255</td></tr>
    <tr><td style="text-align:left">Log Likelihood</td><td>-1,103.131</td></tr>
    <tr><td style="text-align:left">Akaike Inf. Crit.</td><td>2,228.262</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
    </table>
    
    <table style="text-align:center"><tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td>OR</td><td>conf_int_low</td><td>conf_int_high</td><td>p_value</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">(Intercept)</td><td>0.00001</td><td>0.00000</td><td>0.00004</td><td>0</td></tr>
    <tr><td style="text-align:left">n_cancers</td><td>0.639</td><td>0.514</td><td>0.794</td><td>0.039</td></tr>
    <tr><td style="text-align:left">townsend</td><td>1.073</td><td>1.052</td><td>1.094</td><td>0.0003</td></tr>
    <tr><td style="text-align:left">diabetesTRUE</td><td>1.091</td><td>0.902</td><td>1.320</td><td>0.648</td></tr>
    <tr><td style="text-align:left">whr</td><td>5.831</td><td>2.182</td><td>15.581</td><td>0.073</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>1.389</td><td>1.165</td><td>1.658</td><td>0.062</td></tr>
    <tr><td style="text-align:left">age</td><td>1.093</td><td>1.082</td><td>1.105</td><td>0</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>0.760</td><td>0.629</td><td>0.919</td><td>0.148</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>4.172</td><td>2.876</td><td>6.054</td><td>0.0001</td></tr>
    <tr><td style="text-align:left">frontoDem_diagTRUE</td><td>16.361</td><td>5.446</td><td>49.157</td><td>0.011</td></tr>
    <tr><td style="text-align:left">log_crp</td><td>1.108</td><td>1.039</td><td>1.182</td><td>0.111</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr></table>



#### AD model


```r
%%R

ukb_covid_death_covid_AD.b = glm(formula = covid_death ~ 
    age + AD_diag+sex+whr, family = "binomial", 
    data = ukb_dt)

stargazer::stargazer(ukb_covid_death_covid_AD.b, type ="html", 
                     single.row=TRUE,
                     out = "fig_out/ukb_covid_deathAndCOVID_coefficients_ADmodel.html")

ukb_covid_death_covid_AD.b_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid_death_covid_AD.b), 
                           conf_int_low = summary(ukb_covid_death_covid_AD.b)$coefficients[,1] - 
                                          summary(ukb_covid_death_covid_AD.b)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid_death_covid_AD.b)$coefficients[,1] + 
                                          summary(ukb_covid_death_covid_AD.b)$coefficients[,2])),
                 p_value = summary(ukb_covid_death_covid_AD.b)$coefficients[,4]))

stargazer::stargazer(ukb_covid_death_covid_AD.b_odds, type ="html", 
                     single.row=TRUE, summary = FALSE, 
                     out = "fig_out/ukb_covid_deathAndCOVID_odds_ADmodel.html")
write.csv(ukb_covid_death_covid_AD.b_odds,"data_output/ukb_covid_deathAndCOVID_odds_ADmodel.csv", row.names=TRUE)

ukb_covid_death_covid_AD.b_odds$significance = "p-value > 0.05"
ukb_covid_death_covid_AD.b_odds$significance[ukb_covid_death_covid_AD.b_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid_death_covid_AD.b_odds$variables = row.names(ukb_covid_death_covid_AD.b_odds)

library(ggplot2)
ggplot(ukb_covid_death_covid_AD.b_odds, 
       aes(x=reorder(variables, OR), y=OR, color = significance)) + 
    geom_point(shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Mortality odds ratios, death and COVID-19, AD model") + 
  xlab("Variables")

ggsave('fig_out/UKB_deathAndCOVID_ORplot_sig_only_ADmodel.pdf', width = 6, height = 4)
ggsave('fig_out/UKB_deathAndCOVID_ORplot_sig_only_ADmodel.png', width = 6, height = 4)
```


    
    <table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td><em>Dependent variable:</em></td></tr>
    <tr><td></td><td colspan="1" style="border-bottom: 1px solid black"></td></tr>
    <tr><td style="text-align:left"></td><td>covid_death</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">age</td><td>0.667<sup>***</sup> (0.080)</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>1.440<sup>***</sup> (0.349)</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>0.158 (0.156)</td></tr>
    <tr><td style="text-align:left">whr</td><td>0.322<sup>***</sup> (0.077)</td></tr>
    <tr><td style="text-align:left">Constant</td><td>-4.178<sup>***</sup> (0.111)</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>13,252</td></tr>
    <tr><td style="text-align:left">Log Likelihood</td><td>-1,328.771</td></tr>
    <tr><td style="text-align:left">Akaike Inf. Crit.</td><td>2,667.542</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
    </table>
    
    <table style="text-align:center"><tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td>OR</td><td>conf_int_low</td><td>conf_int_high</td><td>p_value</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">(Intercept)</td><td>0.015</td><td>0.014</td><td>0.017</td><td>0</td></tr>
    <tr><td style="text-align:left">age</td><td>1.948</td><td>1.798</td><td>2.110</td><td>0</td></tr>
    <tr><td style="text-align:left">AD_diagTrue</td><td>4.220</td><td>2.978</td><td>5.982</td><td>0.00004</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>1.171</td><td>1.003</td><td>1.368</td><td>0.310</td></tr>
    <tr><td style="text-align:left">whr</td><td>1.381</td><td>1.279</td><td>1.491</td><td>0.00003</td></tr>
    <tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr></table>



#### White matter hypersintensity model


```r
%%R
summary(ukb_covid_and_death.b <-glm(data = ukb_dt, covid_death ~ #Population + n_cancers + smoking + whistling + diabetes+
                                 whr + sex + age +townsend
#                                     +peopleInHousehold+AverageHouseholdIncome
                                    +ethnicity_sim+highBP+
#                           PD_diag+ AD_diag + vascDem_diag+frontoDem_diag+
#                           greyVol 
#                                     + whiteVol 
                                    + AD_diag
#                           edu_score+ crp_norm+
#                           edu_level + house_type+
#                           pm25_val+no2_val+nox_val+so2_val+pm10_val+o3_val
                                    , family = 'binomial'))                                   
```


    
    Call:
    glm(formula = covid_death ~ whr + sex + age + townsend + ethnicity_sim + 
        highBP + +AD_diag, family = "binomial", data = ukb_dt)
    
    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -0.8392  -0.2439  -0.1808  -0.1240   3.3356  
    
    Coefficients:
                         Estimate Std. Error z value Pr(>|z|)    
    (Intercept)        -11.868701   1.034713 -11.471  < 2e-16 ***
    whr                  2.812646   0.845737   3.326 0.000882 ***
    sexMale              0.227886   0.157048   1.451 0.146762    
    age                  0.081450   0.009763   8.342  < 2e-16 ***
    townsend             0.074186   0.017420   4.259 2.06e-05 ***
    ethnicity_simwhite  -0.276915   0.172010  -1.610 0.107425    
    highBPFalse         -0.441129   0.338972  -1.301 0.193131    
    highBPTrue          -0.085265   0.334391  -0.255 0.798735    
    AD_diagTrue          1.435865   0.350335   4.099 4.16e-05 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for binomial family taken to be 1)
    
        Null deviance: 2812.2  on 13229  degrees of freedom
    Residual deviance: 2617.3  on 13221  degrees of freedom
      (108 observations deleted due to missingness)
    AIC: 2635.3
    
    Number of Fisher Scoring iterations: 7
    




```r
%%R
summary(ukb_covid_and_death.b <-glm(data = ukb_dt, covid_death ~ #Population + n_cancers + smoking + whistling + diabetes+
                                 whr + sex + age +townsend+
#                                     +peopleInHousehold+AverageHouseholdIncome
#                                     +ethnicity_sim+highBP+
#                           PD_diag+ AD_diag + vascDem_diag+frontoDem_diag+
#                           greyVol 
#                                     + whiteVol 
                                    log_wmh
#                           edu_score+ crp_norm+
#                           edu_level + house_type+
#                           pm25_val+no2_val+nox_val+so2_val+pm10_val+o3_val
                                    , family = 'binomial'))

```


    
    Call:
    glm(formula = covid_death ~ whr + sex + age + townsend + log_wmh, 
        family = "binomial", data = ukb_dt)
    
    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -0.7815  -0.0728  -0.0183  -0.0037   3.6233  
    
    Coefficients:
                 Estimate Std. Error z value Pr(>|z|)  
    (Intercept) -40.73758   16.53183  -2.464   0.0137 *
    whr          11.18633    7.31094   1.530   0.1260  
    sexMale      -1.56655    1.24340  -1.260   0.2077  
    age           0.39223    0.16850   2.328   0.0199 *
    townsend      0.06657    0.15811   0.421   0.6737  
    log_wmh       0.52021    0.48487   1.073   0.2833  
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for binomial family taken to be 1)
    
        Null deviance: 61.174  on 836  degrees of freedom
    Residual deviance: 42.614  on 831  degrees of freedom
      (12501 observations deleted due to missingness)
    AIC: 54.614
    
    Number of Fisher Scoring iterations: 10
    




```r
%%R
ukb_dt_wmh = na.omit(subset(ukb_dt, select = c(AD_diag, log_wmh, covid_death)))
sum(ukb_dt_wmh$AD_diag==TRUE)
```


    [1] 0



The white matter hyperintensity analysis was not significant because there were only 2 participants who both died and received brain scans. Future work could elucidate this further.

### AD-subset model

The CRP variable was deleted due to NA. Adding it does not change the main conclusion of the model. 


```r
%%R
ukb_dt_AD = subset(ukb_dt, AD_diag == "True")
ukb_covid_death_covid_red.b = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim,
    family = "binomial", data = ukb_dt_AD)

stargazer::stargazer(ukb_covid_death_covid_red.b, type ="html", 
                     single.row=TRUE,
                     out = "fig_out/ukb_deathAndCOVID_coefficients_subsetAD.html")

ukb_covid_death_covid_red.b_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid_death_covid_red.b), 
                           conf_int_low = summary(ukb_covid_death_covid_red.b)$coefficients[,1] - 
                                          summary(ukb_covid_death_covid_red.b)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid_death_covid_red.b)$coefficients[,1] + 
                                          summary(ukb_covid_death_covid_red.b)$coefficients[,2])),
                 p_value = summary(ukb_covid_death_covid_red.b)$coefficients[,4]))

write.csv(ukb_covid_death_covid_red.b_odds,"data_output/ukb_deathAndCOVID_odds_simplifiedModel_subsetAD.csv", row.names=TRUE)

ukb_covid_death_covid_red.b_odds$significance = "p-value > 0.05"
ukb_covid_death_covid_red.b_odds$significance[ukb_covid_death_covid_red.b_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid_death_covid_red.b_odds$variables = row.names(ukb_covid_death_covid_red.b_odds)

library(ggplot2)
ggplot(ukb_covid_death_covid_red.b_odds, 
       aes(x=reorder(variables, OR), y=OR, color = significance)) + 
    geom_point(shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Mortality odds ratios, death and COVID-19, simplified model for AD participants") + 
  xlab("")
ggsave('fig_out/UKB_deathAndCOVID_ORplot_simplifiedModel_subsetAD.png', width = 6, height = 4)
```


    
    <table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td><em>Dependent variable:</em></td></tr>
    <tr><td></td><td colspan="1" style="border-bottom: 1px solid black"></td></tr>
    <tr><td style="text-align:left"></td><td>covid_death</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">n_cancers</td><td>1.453 (1.097)</td></tr>
    <tr><td style="text-align:left">townsend</td><td>-0.096 (0.118)</td></tr>
    <tr><td style="text-align:left">diabetes</td><td>0.250 (0.944)</td></tr>
    <tr><td style="text-align:left">whr</td><td>-8.547 (6.453)</td></tr>
    <tr><td style="text-align:left">sexMale</td><td>1.400 (0.992)</td></tr>
    <tr><td style="text-align:left">age</td><td>0.015 (0.107)</td></tr>
    <tr><td style="text-align:left">ethnicity_simwhite</td><td>15.363 (1,546.944)</td></tr>
    <tr><td style="text-align:left">Constant</td><td>-11.942 (1,546.975)</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>83</td></tr>
    <tr><td style="text-align:left">Log Likelihood</td><td>-27.042</td></tr>
    <tr><td style="text-align:left">Akaike Inf. Crit.</td><td>70.083</td></tr>
    <tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
    </table>



## Add drug-intake data

### Subset the data to decrease processing time


```r
%%R

med_raw = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/data/medications_20003_dt.csv")
med_coding = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/medications_coding4.tsv", 
                     sep = "\t")
```


```r
%%R

drug_list = c("fluoxetine",
"warfarin",
"hydrocortisone",
"amantadine",
"fluvoxamine",
"donepezil hydrochloride",
"galantamine",
"memantine hydrochloride","doxycycline")
coding_subset = med_coding[med_coding$meaning %in% drug_list, ]
# length(drug_list)
nrow(coding_subset)
```


    [1] 9




```r
%%R
#subset 
eid_covid = read.csv("/Users/yizhouyu/camDrive/Miguel_shared_Yizhou/COVID_project_yizhou/AirPoll_UKB_6000/ukb_covid19/data_output/ukb_covid_addDeath.csv")[,c("eid")]

med_covid = med_raw[med_raw$eid %in% eid_covid, ]


```

### Create separate columns for each of the 8 drugs investigated


```r
%%R
coding_subset
```


             coding                 meaning
    2577 1140873394             doxycycline
    2960 1140874896          hydrocortisone
    3509 1140879540              fluoxetine
    3510 1140879544             fluvoxamine
    3523 1140879644              amantadine
    4247 1140888266                warfarin
    5199 1141150834 donepezil hydrochloride
    5828 1141171578             galantamine
    6219 1141182732 memantine hydrochloride




```r
%%R

med_covid$hydrocortisone = apply(med_covid, 1, function(x) any(x %in% c("1140874896")))
med_covid$fluoxetine = apply(med_covid, 1, function(x) any(x %in% c("1140879540")))
med_covid$fluvoxamine = apply(med_covid, 1, function(x) any(x %in% c("1140879544")))
med_covid$amantadine = apply(med_covid, 1, function(x) any(x %in% c("1140879644")))
med_covid$warfarin = apply(med_covid, 1, function(x) any(x %in% c("1140888266")))
med_covid$donepezil = apply(med_covid, 1, function(x) any(x %in% c("1141150834")))
med_covid$galantamine = apply(med_covid, 1, function(x) any(x %in% c("1141171578")))
med_covid$memantine = apply(med_covid, 1, function(x) any(x %in% c("1141182732")))
med_covid$doxycycline = apply(med_covid, 1, function(x) any(x %in% c("1140873394")))                            

```


```r
%%R
med_covid_out = med_covid[,c("eid","hydrocortisone","fluoxetine",
                             "fluvoxamine","amantadine","warfarin","donepezil","galantamine","memantine",
                             "doxycycline")]

write.csv(med_covid_out,"data_output/covid_drug_subset.csv",row.names=FALSE)

```

### Check how many people take each drug in the drug dataset


```r
%%R

colSums(med_covid_out[,-1])
```


    hydrocortisone     fluoxetine    fluvoxamine     amantadine       warfarin 
                36            228              1              9            249 
         donepezil    galantamine      memantine    doxycycline 
                 1              0              0             15 



### Merge and save


```r
%%R
write.csv(merge(
    read.csv(
        "/Users/yizhouyu/camDrive/Miguel_shared_Yizhou/COVID_project_yizhou/AirPoll_UKB_6000/ukb_covid19/data_output/ukb_covid_addDeath.csv"),
    read.csv("data_output/covid_drug_subset.csv"), by = "eid",all.x = TRUE),
          "data_output/ukb_covid_addDrugs.csv", row.names = F)

```

## Relationship between 9 drugs and COVID-19 related death

### Descriptive stats for the drugs


```r
%%R
ukb_dt = read.csv("data_output/ukb_covid_addDrugs.csv")
ukb_dt$covid_death = ifelse(ukb_dt$result == 1 & ukb_dt$death == TRUE, 
1, 0)

library(arsenal)

ukb_descript_stats <- tableby(covid_death ~ ., data = ukb_dt[,2:ncol(ukb_dt)])
write2pdf(ukb_descript_stats, "ukb_descriptStats_drugs.pdf",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data") # passed to summary.tableby
write2word(ukb_descript_stats, "ukb_descriptStats_drugs.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data") # passed to summary.tableby
```

### Subset descriptive statistics for dementia




```r
%%R
ukb_dt = read.csv("data_output/ukb_covid_addDrugs.csv")
ukb_dt$covid_death = ifelse(ukb_dt$result == 1 & ukb_dt$death == TRUE, 
1, 0)
ukb_dt_pos = subset(ukb_dt, result == 1)

ukb_dt_AD = subset(ukb_dt_pos, AD_diag == "True")
ukb_dt_PD = subset(ukb_dt_pos, PD_diag == "True")
ukb_dt_dem = subset(ukb_dt_pos, allDem_diag == TRUE)

library(arsenal)

ukb_descript_COVIDpos <- tableby(covid_death ~ ., data = ukb_dt_pos[,3:ncol(ukb_dt_pos)])
write2word(ukb_descript_COVIDpos, "ukb_descript_COVIDpos.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, all participants were tested positive") # passed to summary.tableby

```


```r
%%R
# infection with drugs

ukb_dt = read.csv("data_output/ukb_covid_addDrugs.csv")
colnames(ukb_dt)
```


     [1] "eid"                    "n_cancers"              "townsend"              
     [4] "smoking"                "ethnicity"              "whistling"             
     [7] "diabetes"               "sex"                    "peopleInHousehold"     
    [10] "AverageHouseholdIncome" "result"                 "PD_diag"               
    [13] "AD_diag"                "highBP"                 "age"                   
    [16] "no2_val"                "nox_val"                "so2_val"               
    [19] "pm10_val"               "o3_val"                 "pm25_val"              
    [22] "Population"             "whr"                    "edu_level"             
    [25] "house_type"             "ethnicity_sim"          "greyVol"               
    [28] "whiteVol"               "brainVol"               "wmh"                   
    [31] "edu_score"              "crp"                    "tot_p"                 
    [34] "all_cause_dem_date"     "all_cause_dem_source"   "AD_date"               
    [37] "AD_source"              "vasc_dem_date"          "vasc_dem_source"       
    [40] "fronto_dem_date"        "fronto_dem_source"      "allDem_diag"           
    [43] "frontoDem_diag"         "vascDem_diag"           "crp_norm"              
    [46] "wmh_norm"               "blood_group"            "log_wmh"               
    [49] "log_crp"                "log_edu"                "death"                 
    [52] "hydrocortisone"         "fluoxetine"             "fluvoxamine"           
    [55] "amantadine"             "warfarin"               "donepezil"             
    [58] "galantamine"            "memantine"              "doxycycline"           




```r
%%R
# infection with drugs

ukb_dt = read.csv("data_output/ukb_covid_addDrugs.csv")
ukb_descript_COVID_infection <- tableby(result ~ ., data = ukb_dt[,c(11,12,13,15,43,44,52:ncol(ukb_dt))])
write2word(ukb_descript_COVID_infection, "ukb_descript_COVID_infectionDrugs.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, drugs & COVID-19 infection") # passed to summary.tableby

ukb_descript_COVID_amantidine <- tableby(amantadine ~ ., data = ukb_dt[,c(11,12,13,15,43,44,52:ncol(ukb_dt))])
write2word(ukb_descript_COVID_amantidine, "ukb_descript_COVID_infectionAmantidine.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, Amantadine") # passed to summary.tableby

```


```r
%%R
ukb_descript_COVIDpos_AD <- tableby(covid_death ~ ., data = ukb_dt_AD[,54:ncol(ukb_dt_AD)])
write2word(ukb_descript_COVIDpos_AD, "ukb_descript_COVIDpos_AD.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, all participants were tested positive and have AD") # passed to summary.tableby

ukb_descript_COVIDpos_PD <- tableby(covid_death ~ ., data = ukb_dt_PD[,54:ncol(ukb_dt_PD)])
write2word(ukb_descript_COVIDpos_PD, "ukb_descript_COVIDpos_PD.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, all participants were tested positive and have PD") # passed to summary.tableby

ukb_descript_COVIDpos_dem <- tableby(covid_death ~ ., data = ukb_dt_dem[,54:ncol(ukb_dt_dem)])
write2word(ukb_descript_COVIDpos_dem, "ukb_descript_COVIDpos_dem.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, all participants were tested positive and have dementia") # passed to summary.tableby

```


```r
%%R
# covid_hydro.b = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
#     whr + sex + age + ethnicity_sim + AD_diag + frontoDem_diag + 
#     log_crp + log_edu+
#                     hydrocortisone,                   
#     family = "binomial", data = ukb_dt)
ukb_dt = read.csv("data_output/ukb_covid_addDrugs.csv")
ukb_dt$covid_death = ifelse(ukb_dt$result == 1 & ukb_dt$death == TRUE, 
1, 0)

ukb_dt_covid = subset(ukb_dt, result == 1)

covid_hydro.b = glm(formula = covid_death ~ sex + age +
                    hydrocortisone,                   
    family = "binomial", data = ukb_dt_covid)

covid_fluo.b = glm(formula = covid_death ~ sex + age +
                   fluoxetine,                   
    family = "binomial", data = ukb_dt_covid)

covid_aman.b = glm(formula = covid_death ~ sex + age +
                   amantadine,                   
    family = "binomial", data = ukb_dt_covid)

covid_war.b = glm(formula = covid_death ~ sex + age +
                   warfarin,                   
    family = "binomial", data = ukb_dt_covid)

covid_dox.b = glm(formula = covid_death ~ sex + age +
                   doxycycline,                   
    family = "binomial", data = ukb_dt_covid)

stargazer::stargazer(covid_hydro.b,covid_fluo.b,covid_aman.b,covid_war.b,covid_dox.b, type ="html", 
                     single.row=TRUE,
                     out = "fig_out/ukb_deathAndCOVID_drug_models.html")
```


    
    <table style="text-align:center"><tr><td colspan="6" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="5"><em>Dependent variable:</em></td></tr>
    <tr><td></td><td colspan="5" style="border-bottom: 1px solid black"></td></tr>
    <tr><td style="text-align:left"></td><td colspan="5">covid_death</td></tr>
    <tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td><td>(3)</td><td>(4)</td><td>(5)</td></tr>
    <tr><td colspan="6" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">sexMale</td><td>0.443<sup>***</sup> (0.141)</td><td>0.443<sup>***</sup> (0.141)</td><td>0.454<sup>***</sup> (0.141)</td><td>0.447<sup>***</sup> (0.141)</td><td>0.449<sup>***</sup> (0.141)</td></tr>
    <tr><td style="text-align:left">age</td><td>0.115<sup>***</sup> (0.010)</td><td>0.115<sup>***</sup> (0.010)</td><td>0.115<sup>***</sup> (0.010)</td><td>0.114<sup>***</sup> (0.010)</td><td>0.115<sup>***</sup> (0.010)</td></tr>
    <tr><td style="text-align:left">hydrocortisone</td><td>1.614 (1.272)</td><td></td><td></td><td></td><td></td></tr>
    <tr><td style="text-align:left">fluoxetine</td><td></td><td>-1.229 (1.043)</td><td></td><td></td><td></td></tr>
    <tr><td style="text-align:left">amantadine</td><td></td><td></td><td>-12.314 (324.744)</td><td></td><td></td></tr>
    <tr><td style="text-align:left">warfarin</td><td></td><td></td><td></td><td>0.344 (0.421)</td><td></td></tr>
    <tr><td style="text-align:left">doxycycline</td><td></td><td></td><td></td><td></td><td>0.395 (1.262)</td></tr>
    <tr><td style="text-align:left">Constant</td><td>-9.894<sup>***</sup> (0.717)</td><td>-9.859<sup>***</sup> (0.715)</td><td>-9.922<sup>***</sup> (0.717)</td><td>-9.873<sup>***</sup> (0.717)</td><td>-9.904<sup>***</sup> (0.716)</td></tr>
    <tr><td colspan="6" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>1,626</td><td>1,626</td><td>1,626</td><td>1,626</td><td>1,626</td></tr>
    <tr><td style="text-align:left">Log Likelihood</td><td>-669.139</td><td>-669.025</td><td>-669.436</td><td>-669.684</td><td>-669.963</td></tr>
    <tr><td style="text-align:left">Akaike Inf. Crit.</td><td>1,346.279</td><td>1,346.049</td><td>1,346.872</td><td>1,347.367</td><td>1,347.926</td></tr>
    <tr><td colspan="6" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td colspan="5" style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
    </table>



### Susbet amantidine


```r
%%R

ukb_dt = read.csv("data_output/ukb_covid_addDrugs.csv")
ukb_dt_amantadine = subset(ukb_dt, amantadine == TRUE)

write.csv(ukb_dt_amantadine, "data_output/ukb_dt_amantadine_susbet.csv")
```

# AD / PD drug analysis

## Data curation

### Subset AD & PD eids


```r
%%R

ukb_dt = read.csv("data_output/ukb_covid_addDrugs.csv")
ad_pd = subset(ukb_dt, AD_diag == TRUE | PD_diag == TRUE)
length(ad_pd$eid)
```


    [1] 239




```r
%%R

med_raw = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/data/medications_20003_dt.csv")
med_coding = read.csv("/Users/yizhouyu/camDrive/ARUK_project/additional_datasets/medications_coding4.tsv", 
                     sep = "\t")

ad_pd_med = med_raw[med_raw$eid %in% ad_pd$eid, ]
```


```r
%%R
ad_pd_med <- ad_pd_med[,colSums(is.na(ad_pd_med))<nrow(ad_pd_med)]
ncol(ad_pd_med)
write.csv(ad_pd_med,"data_output/ukb_adpd_med.csv",row.names = F)

```

### Overview of drugs taken by these participants


```r
%%R

ad_pd_med = read.csv("data_output/ukb_adpd_med.csv")
ad_pd_med_dt = ad_pd_med[,-c(1:4)]
```

Make the dataframe into a single vector


```r
%%R
ad_pd_med_vec = c()

for (rowi in 1:nrow(ad_pd_med_dt)){
    curr_row = as.vector(unlist(ad_pd_med_dt[rowi,]))
    ad_pd_med_vec = c(ad_pd_med_vec,curr_row)
}
```


```r
%%R
ad_pd_med_vec_na = na.omit(ad_pd_med_vec)
length(ad_pd_med_vec_na)
```


    [1] 1147



Count the frequencies of drugs present in this list 


```r
%%R
ad_pd_med_freq = table(ad_pd_med_vec_na)
ad_pd_med_freq[order(-as.numeric(ad_pd_med_freq))]

```


    ad_pd_med_vec_na
    2038460150 1140868226 1140861958      99999 1140865634 1140860806 1140884600 
            64         58         49         34         26         24         21 
    1140871310 1141146234 1141164060 1141188442 1140864752 1141194794 1140928274 
            19         19         18         18         17         17         15 
    1141191044 1140866738 1140921600       1193 1140879802 1140879616 1140882112 
            15         14         13         12         12         11         11 
    1140923346 1140872338 1140879668 1140860696 1140861998 1140909674 1140872420 
            11         10         10          9          9          9          8 
    1140884488 1140888560 1140909708 1140926934 1141164872 1141189134 1140852976 
             8          8          8          8          8          8          7 
    1140879406 1140879644 1140879760 1140888648 1140925800 1141168318 1140874420 
             7          7          7          7          7          7          6 
    1140879540 1140879778 1140888266 1140888552 1140922174 1141176832 1140864992 
             6          6          6          6          6          6          5 
    1140865350 1140865800 1140870422 1140871050 1140874744 1140875408 1140881856 
             5          5          5          5          5          5          5 
    1140883066 1140883476 1141156836 1141169700 1141177526 1140867888 1140871462 
             5          5          5          5          5          4          4 
    1140872228 1140872320 1140883504 1140888646 1140910766 1140916282 1140928338 
             4          4          4          4          4          4          4 
    1141152732 1140861090 1140861806 1140863152 1140865716 1140865752 1140866122 
             4          3          3          3          3          3          3 
    1140869848 1140870390 1140870856 1140871024 1140872072 1140872150 1140875420 
             3          3          3          3          3          3          3 
    1140876592 1140879428 1140879634 1140879842 1140883568 1140888098 1140888510 
             3          3          3          3          3          3          3 
    1140888688 1140909786 1140909806 1140909816 1141162764 1141164068 1141171948 
             3          3          3          3          3          3          3 
    1141192410       1187       1203 1140860954 1140861088 1140861954 1140862572 
             3          2          2          2          2          2          2 
    1140862772 1140863144 1140865396 1140866078 1140866236 1140867878 1140868170 
             2          2          2          2          2          2          2 
    1140868550 1140869034 1140871112 1140871692 1140872200 1140872434 1140874930 
             2          2          2          2          2          2          2 
    1140876404 1140879430 1140882110 1140882374 1140883524 1140884444 1140884672 
             2          2          2          2          2          2          2 
    1140888366 1140909702 1140910832 1140911734 1140923336 1140928916 1141146612 
             2          2          2          2          2          2          2 
    1141150944 1141156846 1141157126 1141163114 1141164086 1141164828 1141167594 
             2          2          2          2          2          2          2 
    1141171932 1141174520 1141175684 1141179992 1141187810 1141192736 1141200322 
             2          2          2          2          2          2          2 
    2038459814       1195 1140851812 1140861120 1140861166 1140861416 1140861884 
             2          1          1          1          1          1          1 
    1140861924 1140862016 1140862260 1140862476 1140862810 1140862952 1140863182 
             1          1          1          1          1          1          1 
    1140863202 1140863552 1140864286 1140864950 1140865010 1140865414 1140865416 
             1          1          1          1          1          1          1 
    1140865426 1140865580 1140865588 1140865654 1140865668 1140866026 1140866116 
             1          1          1          1          1          1          1 
    1140866280 1140866406 1140867304 1140867420 1140867444 1140867504 1140867726 
             1          1          1          1          1          1          1 
    1140867818 1140868120 1140868134 1140868192 1140868258 1140868408 1140868472 
             1          1          1          1          1          1          1 
    1140868848 1140870306 1140870554 1140870788 1140870932 1140870954 1140871168 
             1          1          1          1          1          1          1 
    1140871188 1140871196 1140871266 1140872198 1140872290 1140873474 1140873548 
             1          1          1          1          1          1          1 
    1140873642 1140874138 1140874718 1140874794 1140875452 1140875596 1140875840 
             1          1          1          1          1          1          1 
    1140875990 1140876076 1140876136 1140876312 1140876424 1140877892 1140878512 
             1          1          1          1          1          1          1 
    1140879048 1140879404 1140879464 1140879494 1140879620 1140879648 1140879774 
             1          1          1          1          1          1          1 
    1140879794 1140879854 1140880014 1140881318 1140881334 1140881442 1140881446 
             1          1          1          1          1          1          1 
    1140881472 1140882236 1140882618 1140882694 1140883494 1140883548 1140883560 
             1          1          1          1          1          1          1 
    1140883664 1140884498 1140884516 1140888386 1140888502 1140888538 1140888758 
             1          1          1          1          1          1          1 
    1140909368 1140909480 1140909788 1140911658 1140911698 1140911730 1140911732 
             1          1          1          1          1          1          1 
    1140916790 1140923350 1140923402 1140923670 1140926732 1140926848 1140927086 
             1          1          1          1          1          1          1 
    1140927328 1140929012 1141145660 1141145668 1141146138 1141152098 1141152590 
             1          1          1          1          1          1          1 
    1141152848 1141152998 1141153490 1141156858 1141157252 1141157260 1141157364 
             1          1          1          1          1          1          1 
    1141157398 1141162824 1141165512 1141168326 1141168554 1141168680 1141168822 
             1          1          1          1          1          1          1 
    1141168936 1141168944 1141169666 1141171496 1141171502 1141171646 1141172698 
             1          1          1          1          1          1          1 
    1141172924 1141173342 1141173872 1141173888 1141174032 1141176732 1141180200 
             1          1          1          1          1          1          1 
    1141180212 1141180392 1141181700 1141182628 1141184748 1141187776 1141188146 
             1          1          1          1          1          1          1 
    1141189090 1141189132 1141189680 1141190160 1141192248 1141192740 1141192916 
             1          1          1          1          1          1          1 
    1141193290 1141193346 1141194800 1141199858 1141199940 1141200004 1141200072 
             1          1          1          1          1          1          1 
    1141200708 
             1 



2038460150 - paracetamol - 64<br>
1140868226 - aspirin     - 58<br>
1140861958 - simvastatin - 49<br>
99999      - NA<br>
1140865634 - omeprazole  - 26<br>
1140860806 - ramipril    - 24<br>
1140884600 - metformin   - 21<br>
1140871310 - ibuprofen   - 19<br>
1141146234 - atorvastatin   - 19<br>
1141164060 - pramipexole   - 18<br>
1141188442 - glucosamine   - 18<br>
1140864752 - lansoprazole   - 17<br>
1141194794 - bendroflumethiazide   - 17<br>
1140928274 - ropinirole   - 15<br>
1141191044 - levothyroxine   - 15<br>

### Curate drugs with sufficient data 

I should take away paracetamol, aspirin, ibuprofen as these are common drugs


```r
%%R

ad_pd_med = read.csv("data_output/ukb_adpd_med.csv")

ad_pd_med$simvastatin = apply(ad_pd_med, 1, function(x) any(x %in% c("1140861958")))
ad_pd_med$omeprazole = apply(ad_pd_med, 1, function(x) any(x %in% c("1140865634")))
ad_pd_med$ramipril = apply(ad_pd_med, 1, function(x) any(x %in% c("1140860806")))
ad_pd_med$metformin = apply(ad_pd_med, 1, function(x) any(x %in% c("1140884600")))
ad_pd_med$atorvastatin = apply(ad_pd_med, 1, function(x) any(x %in% c("1141146234")))
ad_pd_med$pramipexole = apply(ad_pd_med, 1, function(x) any(x %in% c("1141164060")))
ad_pd_med$glucosamine = apply(ad_pd_med, 1, function(x) any(x %in% c("1141188442")))
ad_pd_med$lansoprazole = apply(ad_pd_med, 1, function(x) any(x %in% c("1140864752")))
ad_pd_med$bendroflumethiazide = apply(ad_pd_med, 1, function(x) any(x %in% c("1141194794"))) 
ad_pd_med$ropinirole = apply(ad_pd_med, 1, function(x) any(x %in% c("1140928274"))) 
ad_pd_med$levothyroxine = apply(ad_pd_med, 1, function(x) any(x %in% c("1141191044")))
```


```r
%%R

ad_pd_med_subs = subset(ad_pd_med, select = c(eid,simvastatin,omeprazole,ramipril,
                                             metformin,atorvastatin,pramipexole,
                                             glucosamine,lansoprazole,
                                             bendroflumethiazide,ropinirole,levothyroxine))

```


```r
%%R

ad_pd_merge = merge(ad_pd,ad_pd_med_subs,by="eid")
ad_pd_merge$covid_death = ifelse(ad_pd_merge$result == 1 & ad_pd_merge$death == TRUE, 
1, 0)
write.csv(ad_pd_merge, "data_output/ad_pd_drugs_fulldt.csv", row.names=F)
```

## Analysis - death


```r
%%R
ad_pd = read.csv("data_output/ad_pd_drugs_fulldt.csv")
colnames(ad_pd)
```


     [1] "eid"                    "n_cancers"              "townsend"              
     [4] "smoking"                "ethnicity"              "whistling"             
     [7] "diabetes"               "sex"                    "peopleInHousehold"     
    [10] "AverageHouseholdIncome" "result"                 "PD_diag"               
    [13] "AD_diag"                "highBP"                 "age"                   
    [16] "no2_val"                "nox_val"                "so2_val"               
    [19] "pm10_val"               "o3_val"                 "pm25_val"              
    [22] "Population"             "whr"                    "edu_level"             
    [25] "house_type"             "ethnicity_sim"          "greyVol"               
    [28] "whiteVol"               "brainVol"               "wmh"                   
    [31] "edu_score"              "crp"                    "tot_p"                 
    [34] "all_cause_dem_date"     "all_cause_dem_source"   "AD_date"               
    [37] "AD_source"              "vasc_dem_date"          "vasc_dem_source"       
    [40] "fronto_dem_date"        "fronto_dem_source"      "allDem_diag"           
    [43] "frontoDem_diag"         "vascDem_diag"           "crp_norm"              
    [46] "wmh_norm"               "blood_group"            "log_wmh"               
    [49] "log_crp"                "log_edu"                "death"                 
    [52] "hydrocortisone"         "fluoxetine"             "fluvoxamine"           
    [55] "amantadine"             "warfarin"               "donepezil"             
    [58] "galantamine"            "memantine"              "doxycycline"           
    [61] "simvastatin"            "omeprazole"             "ramipril"              
    [64] "metformin"              "atorvastatin"           "pramipexole"           
    [67] "glucosamine"            "lansoprazole"           "bendroflumethiazide"   
    [70] "ropinirole"             "levothyroxine"          "covid_death"           



### Data visualisation

In this dataset, the updated drug data starts from column number 61 and ends on column number 71. Drug names: 
simvastatin,omeprazole,ramipril,                                             metformin,atorvastatin,pramipexole,                                             glucosamine,lansoprazole,                                             bendroflumethiazide,ropinirole,levothyroxine


```r
%%R
library(arsenal)
ad_med_descrip = subset(ad_pd, AD_diag == TRUE, select = c(covid_death,simvastatin,omeprazole,ramipril,
                                             metformin,atorvastatin,pramipexole,
                                             glucosamine,lansoprazole,
                                             bendroflumethiazide,ropinirole,levothyroxine))

pd_med_descrip = subset(ad_pd, PD_diag == TRUE, select = c(covid_death,simvastatin,omeprazole,ramipril,
                                             metformin,atorvastatin,pramipexole,
                                             glucosamine,lansoprazole,
                                             bendroflumethiazide,ropinirole,levothyroxine))

ad_med_descrip_t <- tableby(covid_death ~ ., data = ad_med_descrip)
pd_med_descrip_t <- tableby(covid_death ~ ., data = pd_med_descrip)

write2word(ad_med_descrip_t, "ad_drugs_ukb_descriptStats_deaths.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, drugs in AD patients, 1 = death from covid") # passed to summary.tableby

write2word(pd_med_descrip_t, "pd_drugs_ukb_descriptStats_deaths.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, drugs in PD patients, 1 = death from covid") # passed to summary.tableby


```

In this dataset, the updated drug data starts from column number 61 and ends on column number 71. Drug names: 
simvastatin,omeprazole,ramipril,                                             metformin,atorvastatin,pramipexole,                                             glucosamine,lansoprazole,                                             bendroflumethiazide,ropinirole,levothyroxine

I will do separate models for AD vs PD

### AD drug model

#### Complex model


```r
%%R
# AD only dataset
ad_drugs_dt = subset(ad_pd, AD_diag == TRUE)

colnames(ad_drugs_dt)
```


     [1] "eid"                    "n_cancers"              "townsend"              
     [4] "smoking"                "ethnicity"              "whistling"             
     [7] "diabetes"               "sex"                    "peopleInHousehold"     
    [10] "AverageHouseholdIncome" "result"                 "PD_diag"               
    [13] "AD_diag"                "highBP"                 "age"                   
    [16] "no2_val"                "nox_val"                "so2_val"               
    [19] "pm10_val"               "o3_val"                 "pm25_val"              
    [22] "Population"             "whr"                    "edu_level"             
    [25] "house_type"             "ethnicity_sim"          "greyVol"               
    [28] "whiteVol"               "brainVol"               "wmh"                   
    [31] "edu_score"              "crp"                    "tot_p"                 
    [34] "all_cause_dem_date"     "all_cause_dem_source"   "AD_date"               
    [37] "AD_source"              "vasc_dem_date"          "vasc_dem_source"       
    [40] "fronto_dem_date"        "fronto_dem_source"      "allDem_diag"           
    [43] "frontoDem_diag"         "vascDem_diag"           "crp_norm"              
    [46] "wmh_norm"               "blood_group"            "log_wmh"               
    [49] "log_crp"                "log_edu"                "death"                 
    [52] "hydrocortisone"         "fluoxetine"             "fluvoxamine"           
    [55] "amantadine"             "warfarin"               "donepezil"             
    [58] "galantamine"            "memantine"              "doxycycline"           
    [61] "simvastatin"            "omeprazole"             "ramipril"              
    [64] "metformin"              "atorvastatin"           "pramipexole"           
    [67] "glucosamine"            "lansoprazole"           "bendroflumethiazide"   
    [70] "ropinirole"             "levothyroxine"          "covid_death"           




```r
%%R
simvastatin = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    simvastatin,
    family = "binomial", data = ad_drugs_dt)
omeprazole = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    omeprazole,
    family = "binomial", data = ad_drugs_dt)
ramipril = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    ramipril,
    family = "binomial", data = ad_drugs_dt)
metformin = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    metformin,
    family = "binomial", data = ad_drugs_dt)
atorvastatin = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    atorvastatin,
    family = "binomial", data = ad_drugs_dt)
# pramipexole = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
#     whr + sex + age + ethnicity_sim + 
#     log_crp +
#     pramipexole,
#     family = "binomial", data = ad_drugs_dt)
glucosamine = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    glucosamine,
    family = "binomial", data = ad_drugs_dt)
lansoprazole = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    lansoprazole,
    family = "binomial", data = ad_drugs_dt)
bendroflumethiazide = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    bendroflumethiazide,
    family = "binomial", data = ad_drugs_dt)
# ropinirole = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
#     whr + sex + age + ethnicity_sim + 
#     log_crp +
#     ropinirole,
#     family = "binomial", data = ad_drugs_dt)
levothyroxine = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    levothyroxine,
    family = "binomial", data = ad_drugs_dt)

AD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
#                                   "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
#                                   "ropinirole",
                                  "levothyroxine"),
                         OR = c(summary(simvastatin)$coefficients[10,1],
                                 summary(omeprazole)$coefficients[10,1],
                                 summary(ramipril)$coefficients[10,1],
                                 summary(metformin)$coefficients[10,1],
                                 summary(atorvastatin)$coefficients[10,1],
#                                  summary(pramipexole)$coefficients[10,1],
                                 summary(glucosamine)$coefficients[10,1],
                                 summary(lansoprazole)$coefficients[10,1],
                                 summary(bendroflumethiazide)$coefficients[10,1],
#                                  summary(ropinirole)$coefficients[10,1],
                                 summary(levothyroxine)$coefficients[10,1]),
                        p_val = c(summary(simvastatin)$coefficients[10,4],
                                 summary(omeprazole)$coefficients[10,4],
                                 summary(ramipril)$coefficients[10,4],
                                 summary(metformin)$coefficients[10,4],
                                 summary(atorvastatin)$coefficients[10,4],
#                                  summary(pramipexole)$coefficients[10,4],
                                 summary(glucosamine)$coefficients[10,4],
                                 summary(lansoprazole)$coefficients[10,4],
                                 summary(bendroflumethiazide)$coefficients[10,4],
#                                  summary(ropinirole)$coefficients[10,4],
                                 summary(levothyroxine)$coefficients[10,4]))
AD_drugs_df
```


                     drug          OR     p_val
    1         simvastatin  -1.3623323 0.2611871
    2          omeprazole   1.6841169 0.2615778
    3            ramipril   1.1253548 0.4517741
    4           metformin  17.4057213 0.9946937
    5        atorvastatin   1.7336503 0.1490702
    6         glucosamine   1.4611078 0.2085539
    7        lansoprazole -15.5097862 0.9966842
    8 bendroflumethiazide -16.9287283 0.9948159
    9       levothyroxine   0.9282706 0.4837088




```r
%%R
write.csv(AD_drugs_df,"data_output/AD_drugs_models_full.csv",row.names=F)

```

#### Models with only age, sex and whr as covariates


```r
%%R

simvastatin = glm(formula = covid_death ~ whr + sex + age +
    simvastatin,
    family = "binomial", data = ad_drugs_dt)

omeprazole = glm(formula = covid_death ~ whr + sex + age +
    omeprazole,
    family = "binomial", data = ad_drugs_dt)

ramipril = glm(formula = covid_death ~ whr + sex + age +
    ramipril,
    family = "binomial", data = ad_drugs_dt)

metformin = glm(formula = covid_death ~ whr + sex + age +
    metformin,
    family = "binomial", data = ad_drugs_dt)

atorvastatin = glm(formula = covid_death ~ whr + sex + age +
    atorvastatin,
    family = "binomial", data = ad_drugs_dt)

pramipexole = glm(formula = covid_death ~ whr + sex + age +
    pramipexole,
    family = "binomial", data = ad_drugs_dt)

glucosamine = glm(formula = covid_death ~ whr + sex + age +
    glucosamine,
    family = "binomial", data = ad_drugs_dt)

lansoprazole = glm(formula = covid_death ~ whr + sex + age +
    lansoprazole,
    family = "binomial", data = ad_drugs_dt)

bendroflumethiazide = glm(formula = covid_death ~ whr + sex + age +
    bendroflumethiazide,
    family = "binomial", data = ad_drugs_dt)

ropinirole = glm(formula = covid_death ~ whr + sex + age + 
    ropinirole,
    family = "binomial", data = ad_drugs_dt)

levothyroxine = glm(formula = covid_death ~ whr + sex + age +
    levothyroxine,
    family = "binomial", data = ad_drugs_dt)

AD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
#                                   "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
#                                   "ropinirole",
                                  "levothyroxine"),
                         OR = c(summary(simvastatin)$coefficients[5,1],
                                 summary(omeprazole)$coefficients[5,1],
                                 summary(ramipril)$coefficients[5,1],
                                 summary(metformin)$coefficients[5,1],
                                 summary(atorvastatin)$coefficients[5,1],
#                                  summary(pramipexole)$coefficients[5,1],
                                 summary(glucosamine)$coefficients[5,1],
                                 summary(lansoprazole)$coefficients[5,1],
                                 summary(bendroflumethiazide)$coefficients[5,1],
#                                  summary(ropinirole)$coefficients[5,1],
                                 summary(levothyroxine)$coefficients[5,1]),
                        p_val = c(summary(simvastatin)$coefficients[5,4],
                                 summary(omeprazole)$coefficients[5,4],
                                 summary(ramipril)$coefficients[5,4],
                                 summary(metformin)$coefficients[5,4],
                                 summary(atorvastatin)$coefficients[5,4],
#                                  summary(pramipexole)$coefficients[5,4],
                                 summary(glucosamine)$coefficients[5,4],
                                 summary(lansoprazole)$coefficients[5,4],
                                 summary(bendroflumethiazide)$coefficients[5,4],
#                                  summary(ropinirole)$coefficients[5,4],
                                 summary(levothyroxine)$coefficients[5,4]))
AD_drugs_df
```


                     drug          OR     p_val
    1         simvastatin  -1.2312245 0.2742804
    2          omeprazole   0.9135951 0.4585437
    3            ramipril  -0.2833312 0.8075671
    4           metformin   0.6881912 0.4810317
    5        atorvastatin   0.5498309 0.5564945
    6         glucosamine   1.0631918 0.2631753
    7        lansoprazole -15.8980908 0.9933369
    8 bendroflumethiazide -16.4780011 0.9945993
    9       levothyroxine   1.0232247 0.4181616




```r
%%R
write.csv(AD_drugs_df,"data_output/AD_drugs_models_simplified.csv",row.names=F)

```

#### raw odds of drugs in the AD models


```r
%%R

simvastatin = glm(formula = covid_death ~# whr + sex + age +
    simvastatin,
    family = "binomial", data = ad_drugs_dt)

omeprazole = glm(formula = covid_death ~ #whr + sex + age +
    omeprazole,
    family = "binomial", data = ad_drugs_dt)

ramipril = glm(formula = covid_death ~ #whr + sex + age +
    ramipril,
    family = "binomial", data = ad_drugs_dt)

metformin = glm(formula = covid_death ~ #whr + sex + age +
    metformin,
    family = "binomial", data = ad_drugs_dt)

atorvastatin = glm(formula = covid_death ~ #whr + sex + age +
    atorvastatin,
    family = "binomial", data = ad_drugs_dt)

pramipexole = glm(formula = covid_death ~ #whr + sex + age +
    pramipexole,
    family = "binomial", data = ad_drugs_dt)

glucosamine = glm(formula = covid_death ~ #whr + sex + age +
    glucosamine,
    family = "binomial", data = ad_drugs_dt)

lansoprazole = glm(formula = covid_death ~ #whr + sex + age +
    lansoprazole,
    family = "binomial", data = ad_drugs_dt)

bendroflumethiazide = glm(formula = covid_death ~ #whr + sex + age +
    bendroflumethiazide,
    family = "binomial", data = ad_drugs_dt)

ropinirole = glm(formula = covid_death ~ #whr + sex + age + 
    ropinirole,
    family = "binomial", data = ad_drugs_dt)

levothyroxine = glm(formula = covid_death ~ #whr + sex + age +
    levothyroxine,
    family = "binomial", data = ad_drugs_dt)

AD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
#                                   "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
#                                   "ropinirole",
                                  "levothyroxine"),
                         OR = c(summary(simvastatin)$coefficients[2,1],
                                 summary(omeprazole)$coefficients[2,1],
                                 summary(ramipril)$coefficients[2,1],
                                 summary(metformin)$coefficients[2,1],
                                 summary(atorvastatin)$coefficients[2,1],
#                                  summary(pramipexole)$coefficients[2,1],
                                 summary(glucosamine)$coefficients[2,1],
                                 summary(lansoprazole)$coefficients[2,1],
                                 summary(bendroflumethiazide)$coefficients[2,1],
#                                  summary(ropinirole)$coefficients[2,1],
                                 summary(levothyroxine)$coefficients[2,1]),
                        p_val = c(summary(simvastatin)$coefficients[2,4],
                                 summary(omeprazole)$coefficients[2,4],
                                 summary(ramipril)$coefficients[2,4],
                                 summary(metformin)$coefficients[2,4],
                                 summary(atorvastatin)$coefficients[2,4],
#                                  summary(pramipexole)$coefficients[2,4],
                                 summary(glucosamine)$coefficients[2,4],
                                 summary(lansoprazole)$coefficients[2,4],
                                 summary(bendroflumethiazide)$coefficients[2,4],
#                                  summary(ropinirole)$coefficients[2,4],
                                 summary(levothyroxine)$coefficients[2,4]))
AD_drugs_df
```


                     drug          OR     p_val
    1         simvastatin  -0.9873867 0.3648760
    2          omeprazole   0.4274440 0.7104472
    3            ramipril  -0.2200619 0.8431803
    4           metformin   0.3589451 0.6747753
    5        atorvastatin   0.5908683 0.4954917
    6         glucosamine   1.2383742 0.1765039
    7        lansoprazole -15.6201583 0.9936995
    8 bendroflumethiazide -15.6639610 0.9916419
    9       levothyroxine   0.2305237 0.8393149




```r
%%R
write.csv(AD_drugs_df,"data_output/AD_drugs_models_onlyDrug.csv",row.names=F)

```

### PD drug models

#### Complex model


```r
%%R
pd_drugs_dt = subset(ad_pd, PD_diag == TRUE)

```


```r
%%R

simvastatin = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    simvastatin,
    family = "binomial", data = pd_drugs_dt)
omeprazole = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    omeprazole,
    family = "binomial", data = pd_drugs_dt)
ramipril = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    ramipril,
    family = "binomial", data = pd_drugs_dt)
metformin = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    metformin,
    family = "binomial", data = pd_drugs_dt)
atorvastatin = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    atorvastatin,
    family = "binomial", data = pd_drugs_dt)
pramipexole = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    pramipexole,
    family = "binomial", data = pd_drugs_dt)
glucosamine = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    glucosamine,
    family = "binomial", data = pd_drugs_dt)
lansoprazole = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    lansoprazole,
    family = "binomial", data = pd_drugs_dt)
bendroflumethiazide = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    bendroflumethiazide,
    family = "binomial", data = pd_drugs_dt)
ropinirole = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    ropinirole,
    family = "binomial", data = pd_drugs_dt)
levothyroxine = glm(formula = covid_death ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    levothyroxine,
    family = "binomial", data = pd_drugs_dt)


PD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
                                  "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
                                  "ropinirole",
                                  "levothyroxine"),
                        OR = c(summary(simvastatin)$coefficients[10,1],
                                 summary(omeprazole)$coefficients[10,1],
                                 summary(ramipril)$coefficients[10,1],
                                 summary(metformin)$coefficients[10,1],
                                 summary(atorvastatin)$coefficients[10,1],
                                 summary(pramipexole)$coefficients[10,1],
                                 summary(glucosamine)$coefficients[10,1],
                                 summary(lansoprazole)$coefficients[10,1],
                                 summary(bendroflumethiazide)$coefficients[10,1],
                                 summary(ropinirole)$coefficients[10,1],
                                 summary(levothyroxine)$coefficients[10,1]),
                        p_val = c(summary(simvastatin)$coefficients[10,4],
                                 summary(omeprazole)$coefficients[10,4],
                                 summary(ramipril)$coefficients[10,4],
                                 summary(metformin)$coefficients[10,4],
                                 summary(atorvastatin)$coefficients[10,4],
                                 summary(pramipexole)$coefficients[10,4],
                                 summary(glucosamine)$coefficients[10,4],
                                 summary(lansoprazole)$coefficients[10,4],
                                 summary(bendroflumethiazide)$coefficients[10,4],
                                 summary(ropinirole)$coefficients[10,4],
                                 summary(levothyroxine)$coefficients[10,4]))
PD_drugs_df

```


                      drug          OR     p_val
    1          simvastatin   0.4658756 0.7674519
    2           omeprazole   1.2807060 0.5388554
    3             ramipril -18.2709600 0.9964870
    4            metformin  -0.7207236 0.9999425
    5         atorvastatin -12.6315602 0.9982307
    6          pramipexole -16.7533019 0.9963677
    7          glucosamine -16.7480751 0.9972084
    8         lansoprazole -17.7079015 0.9967769
    9  bendroflumethiazide   0.8652933 0.5511678
    10          ropinirole   2.2390895 0.1834277
    11       levothyroxine  20.0595389 0.9955084




```r
%%R
write.csv(PD_drugs_df,"data_output/PD_drugs_models_full.csv",row.names=F)

```

I went through the drugs one by one here, but did not find any significance in any of these drugs.

#### Simplified model


```r
%%R

simvastatin = glm(formula = covid_death ~ whr + sex + age +
    simvastatin,
    family = "binomial", data = pd_drugs_dt)
omeprazole = glm(formula = covid_death ~ whr + sex + age +
    omeprazole,
    family = "binomial", data = pd_drugs_dt)
ramipril = glm(formula = covid_death ~ whr + sex + age +
    ramipril,
    family = "binomial", data = pd_drugs_dt)
metformin = glm(formula = covid_death ~ whr + sex + age +
    metformin,
    family = "binomial", data = pd_drugs_dt)
atorvastatin = glm(formula = covid_death ~ whr + sex + age +
    atorvastatin,
    family = "binomial", data = pd_drugs_dt)
pramipexole = glm(formula = covid_death ~ whr + sex + age +
    pramipexole,
    family = "binomial", data = pd_drugs_dt)
glucosamine = glm(formula = covid_death ~ whr + sex + age +
    glucosamine,
    family = "binomial", data = pd_drugs_dt)
lansoprazole = glm(formula = covid_death ~ whr + sex + age +
    lansoprazole,
    family = "binomial", data = pd_drugs_dt)
bendroflumethiazide = glm(formula = covid_death ~ whr + sex + age +
    bendroflumethiazide,
    family = "binomial", data = pd_drugs_dt)
ropinirole = glm(formula = covid_death ~ whr + sex + age +
    ropinirole,
    family = "binomial", data = pd_drugs_dt)
levothyroxine = glm(formula = covid_death ~ whr + sex + age +
    levothyroxine,
    family = "binomial", data = pd_drugs_dt)


PD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
                                  "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
                                  "ropinirole",
                                  "levothyroxine"),
                         OR = c(summary(simvastatin)$coefficients[5,1],
                                 summary(omeprazole)$coefficients[5,1],
                                 summary(ramipril)$coefficients[5,1],
                                 summary(metformin)$coefficients[5,1],
                                 summary(atorvastatin)$coefficients[5,1],
                                 summary(pramipexole)$coefficients[5,1],
                                 summary(glucosamine)$coefficients[5,1],
                                 summary(lansoprazole)$coefficients[5,1],
                                 summary(bendroflumethiazide)$coefficients[5,1],
                                 summary(ropinirole)$coefficients[5,1],
                                 summary(levothyroxine)$coefficients[5,1]),
                        p_val = c(summary(simvastatin)$coefficients[5,4],
                                 summary(omeprazole)$coefficients[5,4],
                                 summary(ramipril)$coefficients[5,4],
                                 summary(metformin)$coefficients[5,4],
                                 summary(atorvastatin)$coefficients[5,4],
                                 summary(pramipexole)$coefficients[5,4],
                                 summary(glucosamine)$coefficients[5,4],
                                 summary(lansoprazole)$coefficients[5,4],
                                 summary(bendroflumethiazide)$coefficients[5,4],
                                 summary(ropinirole)$coefficients[5,4],
                                 summary(levothyroxine)$coefficients[5,4]))
PD_drugs_df

```


                      drug           OR      p_val
    1          simvastatin   0.07162034 0.95304938
    2           omeprazole   1.74720240 0.09091277
    3             ramipril -16.50258464 0.99512349
    4            metformin -14.88151427 0.99361670
    5         atorvastatin -14.11293706 0.99438294
    6          pramipexole -16.46162860 0.99438476
    7          glucosamine -14.93117251 0.99368414
    8         lansoprazole -15.91273856 0.99531145
    9  bendroflumethiazide   0.89648212 0.46777092
    10          ropinirole   1.43610977 0.26555036
    11       levothyroxine   1.65734261 0.28396684




```r
%%R
summary(omeprazole)
```


    
    Call:
    glm(formula = covid_death ~ whr + sex + age + omeprazole, family = "binomial", 
        data = pd_drugs_dt)
    
    Deviance Residuals: 
         Min        1Q    Median        3Q       Max  
    -0.97748  -0.27678  -0.16898  -0.09128   2.92397  
    
    Coefficients:
                   Estimate Std. Error z value Pr(>|z|)  
    (Intercept)     -7.1369    12.1535  -0.587   0.5570  
    whr            -14.0625     6.9706  -2.017   0.0437 *
    sexMale          1.6665     1.1387   1.464   0.1433  
    age              0.1955     0.1369   1.428   0.1532  
    omeprazoleTRUE   1.7472     1.0335   1.691   0.0909 .
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for binomial family taken to be 1)
    
        Null deviance: 50.785  on 154  degrees of freedom
    Residual deviance: 39.988  on 150  degrees of freedom
      (2 observations deleted due to missingness)
    AIC: 49.988
    
    Number of Fisher Scoring iterations: 7
    




```r
%%R
write.csv(PD_drugs_df,"data_output/PD_drugs_models_simplified.csv",row.names=F)

```

#### Model with only drug - raw odds


```r
%%R

simvastatin = glm(formula = covid_death ~ #whr + sex + age +
    simvastatin,
    family = "binomial", data = pd_drugs_dt)
omeprazole = glm(formula = covid_death ~ #whr + sex + age +
    omeprazole,
    family = "binomial", data = pd_drugs_dt)
ramipril = glm(formula = covid_death ~ #whr + sex + age +
    ramipril,
    family = "binomial", data = pd_drugs_dt)
metformin = glm(formula = covid_death ~ #whr + sex + age +
    metformin,
    family = "binomial", data = pd_drugs_dt)
atorvastatin = glm(formula = covid_death ~ #whr + sex + age +
    atorvastatin,
    family = "binomial", data = pd_drugs_dt)
pramipexole = glm(formula = covid_death ~ #whr + sex + age +
    pramipexole,
    family = "binomial", data = pd_drugs_dt)
glucosamine = glm(formula = covid_death ~ #whr + sex + age +
    glucosamine,
    family = "binomial", data = pd_drugs_dt)
lansoprazole = glm(formula = covid_death ~ #whr + sex + age +
    lansoprazole,
    family = "binomial", data = pd_drugs_dt)
bendroflumethiazide = glm(formula = covid_death ~ #whr + sex + age +
    bendroflumethiazide,
    family = "binomial", data = pd_drugs_dt)
ropinirole = glm(formula = covid_death ~ #whr + sex + age +
    ropinirole,
    family = "binomial", data = pd_drugs_dt)
levothyroxine = glm(formula = covid_death ~ #whr + sex + age +
    levothyroxine,
    family = "binomial", data = pd_drugs_dt)


PD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
                                  "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
                                  "ropinirole",
                                  "levothyroxine"),
                         OR = c(summary(simvastatin)$coefficients[2,1],
                                 summary(omeprazole)$coefficients[2,1],
                                 summary(ramipril)$coefficients[2,1],
                                 summary(metformin)$coefficients[2,1],
                                 summary(atorvastatin)$coefficients[2,1],
                                 summary(pramipexole)$coefficients[2,1],
                                 summary(glucosamine)$coefficients[2,1],
                                 summary(lansoprazole)$coefficients[2,1],
                                 summary(bendroflumethiazide)$coefficients[2,1],
                                 summary(ropinirole)$coefficients[2,1],
                                 summary(levothyroxine)$coefficients[2,1]),
                        p_val = c(summary(simvastatin)$coefficients[2,4],
                                 summary(omeprazole)$coefficients[2,4],
                                 summary(ramipril)$coefficients[2,4],
                                 summary(metformin)$coefficients[2,4],
                                 summary(atorvastatin)$coefficients[2,4],
                                 summary(pramipexole)$coefficients[2,4],
                                 summary(glucosamine)$coefficients[2,4],
                                 summary(lansoprazole)$coefficients[2,4],
                                 summary(bendroflumethiazide)$coefficients[2,4],
                                 summary(ropinirole)$coefficients[2,4],
                                 summary(levothyroxine)$coefficients[2,4]))
PD_drugs_df

```


                      drug          OR     p_val
    1          simvastatin  -0.2148447 0.8471060
    2           omeprazole   1.2452158 0.1665432
    3             ramipril -15.4378471 0.9929342
    4            metformin -15.3949833 0.9946736
    5         atorvastatin -15.4020009 0.9943479
    6          pramipexole -16.4674789 0.9948164
    7          glucosamine -15.4161856 0.9937456
    8         lansoprazole -15.4305743 0.9931944
    9  bendroflumethiazide   1.1491646 0.3168656
    10          ropinirole   0.8401294 0.4595782
    11       levothyroxine   1.4144652 0.2234264




```r
%%R
write.csv(PD_drugs_df,"data_output/PD_drugs_models_drugsOnly.csv",row.names=F)

```

## Analysis - infection

### AD infection models

#### Visualisation


```r
%%R
ad_pd = read.csv("data_output/ad_pd_drugs_fulldt.csv")
library(arsenal)
pd_med_descrip_cases = subset(ad_pd, PD_diag == TRUE, select = c(result,simvastatin,omeprazole,ramipril,
                                             metformin,atorvastatin,pramipexole,
                                             glucosamine,lansoprazole,
                                             bendroflumethiazide,ropinirole,levothyroxine))

ad_med_descrip_cases = subset(ad_pd, AD_diag == TRUE, select = c(result,simvastatin,omeprazole,ramipril,
                                             metformin,atorvastatin,pramipexole,
                                             glucosamine,lansoprazole,
                                             bendroflumethiazide,ropinirole,levothyroxine))

pd_med_descrip_t <- tableby(result ~ ., data = pd_med_descrip_cases)
ad_med_descrip_t <- tableby(result ~ ., data = ad_med_descrip_cases)

write2word(ad_med_descrip_t, "ad_drugs_ukb_descriptStats_cases.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, drugs in AD patients, COVID-19 positive = 1") # passed to summary.tableby

write2word(pd_med_descrip_t, "pd_drugs_ukb_descriptStats_cases.doc",
  keep.md = TRUE,
  quiet = TRUE, # passed to rmarkdown::render
  title = "Descriptive statistics of the UK Biobank data, drugs in PD patients, COVID-19 positive = 1") # passed to summary.tableby

```

#### Complex model


```r
%%R

ad_drugs_dt = subset(ad_pd, AD_diag == TRUE)

simvastatin = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    simvastatin,
    family = "binomial", data = ad_drugs_dt)
omeprazole = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    omeprazole,
    family = "binomial", data = ad_drugs_dt)
ramipril = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    ramipril,
    family = "binomial", data = ad_drugs_dt)
metformin = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    metformin,
    family = "binomial", data = ad_drugs_dt)
atorvastatin = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    atorvastatin,
    family = "binomial", data = ad_drugs_dt)
# pramipexole = glm(formula = result ~ n_cancers + townsend + diabetes + 
#     whr + sex + age + ethnicity_sim + 
#     log_crp +
#     pramipexole,
#     family = "binomial", data = ad_drugs_dt)
glucosamine = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    glucosamine,
    family = "binomial", data = ad_drugs_dt)
lansoprazole = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    lansoprazole,
    family = "binomial", data = ad_drugs_dt)
bendroflumethiazide = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    bendroflumethiazide,
    family = "binomial", data = ad_drugs_dt)
# ropinirole = glm(formula = result ~ n_cancers + townsend + diabetes + 
#     whr + sex + age + ethnicity_sim + 
#     log_crp +
#     ropinirole,
#     family = "binomial", data = ad_drugs_dt)
levothyroxine = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    levothyroxine,
    family = "binomial", data = ad_drugs_dt)

AD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
#                                   "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
#                                   "ropinirole",
                                  "levothyroxine"),
                         OR = c(summary(simvastatin)$coefficients[10,1],
                                 summary(omeprazole)$coefficients[10,1],
                                 summary(ramipril)$coefficients[10,1],
                                 summary(metformin)$coefficients[10,1],
                                 summary(atorvastatin)$coefficients[10,1],
#                                  summary(pramipexole)$coefficients[10,1],
                                 summary(glucosamine)$coefficients[10,1],
                                 summary(lansoprazole)$coefficients[10,1],
                                 summary(bendroflumethiazide)$coefficients[10,1],
#                                  summary(ropinirole)$coefficients[10,1],
                                 summary(levothyroxine)$coefficients[10,1]),
                        p_val = c(summary(simvastatin)$coefficients[10,4],
                                 summary(omeprazole)$coefficients[10,4],
                                 summary(ramipril)$coefficients[10,4],
                                 summary(metformin)$coefficients[10,4],
                                 summary(atorvastatin)$coefficients[10,4],
#                                  summary(pramipexole)$coefficients[10,4],
                                 summary(glucosamine)$coefficients[10,4],
                                 summary(lansoprazole)$coefficients[10,4],
                                 summary(bendroflumethiazide)$coefficients[10,4],
#                                  summary(ropinirole)$coefficients[10,4],
                                 summary(levothyroxine)$coefficients[10,4]))
AD_drugs_df
```


                     drug           OR      p_val
    1         simvastatin   0.36144397 0.61416383
    2          omeprazole   0.67235271 0.62079398
    3            ramipril   0.76086798 0.49491516
    4           metformin  -0.04267783 0.97664534
    5        atorvastatin   1.79817407 0.06132538
    6         glucosamine   0.79224227 0.45546781
    7        lansoprazole -16.17441979 0.99422477
    8 bendroflumethiazide  -1.14082976 0.34534770
    9       levothyroxine  -0.77575472 0.52331018



#### Simplified model


```r
%%R

simvastatin = glm(formula = result ~ whr + sex + age +
    simvastatin,
    family = "binomial", data = ad_drugs_dt)

omeprazole = glm(formula = result ~ whr + sex + age +
    omeprazole,
    family = "binomial", data = ad_drugs_dt)

ramipril = glm(formula = result ~ whr + sex + age +
    ramipril,
    family = "binomial", data = ad_drugs_dt)

metformin = glm(formula = result ~ whr + sex + age +
    metformin,
    family = "binomial", data = ad_drugs_dt)

atorvastatin = glm(formula = result ~ whr + sex + age +
    atorvastatin,
    family = "binomial", data = ad_drugs_dt)

pramipexole = glm(formula = result ~ whr + sex + age +
    pramipexole,
    family = "binomial", data = ad_drugs_dt)

glucosamine = glm(formula = result ~ whr + sex + age +
    glucosamine,
    family = "binomial", data = ad_drugs_dt)

lansoprazole = glm(formula = result ~ whr + sex + age +
    lansoprazole,
    family = "binomial", data = ad_drugs_dt)

bendroflumethiazide = glm(formula = result ~ whr + sex + age +
    bendroflumethiazide,
    family = "binomial", data = ad_drugs_dt)

ropinirole = glm(formula = result ~ whr + sex + age + 
    ropinirole,
    family = "binomial", data = ad_drugs_dt)

levothyroxine = glm(formula = result ~ whr + sex + age +
    levothyroxine,
    family = "binomial", data = ad_drugs_dt)

AD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
#                                   "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
#                                   "ropinirole",
                                  "levothyroxine"),
                         OR = c(summary(simvastatin)$coefficients[5,1],
                                 summary(omeprazole)$coefficients[5,1],
                                 summary(ramipril)$coefficients[5,1],
                                 summary(metformin)$coefficients[5,1],
                                 summary(atorvastatin)$coefficients[5,1],
#                                  summary(pramipexole)$coefficients[5,1],
                                 summary(glucosamine)$coefficients[5,1],
                                 summary(lansoprazole)$coefficients[5,1],
                                 summary(bendroflumethiazide)$coefficients[5,1],
#                                  summary(ropinirole)$coefficients[5,1],
                                 summary(levothyroxine)$coefficients[5,1]),
                        p_val = c(summary(simvastatin)$coefficients[5,4],
                                 summary(omeprazole)$coefficients[5,4],
                                 summary(ramipril)$coefficients[5,4],
                                 summary(metformin)$coefficients[5,4],
                                 summary(atorvastatin)$coefficients[5,4],
#                                  summary(pramipexole)$coefficients[5,4],
                                 summary(glucosamine)$coefficients[5,4],
                                 summary(lansoprazole)$coefficients[5,4],
                                 summary(bendroflumethiazide)$coefficients[5,4],
#                                  summary(ropinirole)$coefficients[5,4],
                                 summary(levothyroxine)$coefficients[5,4]))
AD_drugs_df
```


                     drug            OR     p_val
    1         simvastatin  -0.094957715 0.8764135
    2          omeprazole   0.322974403 0.7303070
    3            ramipril  -0.198887226 0.7973555
    4           metformin  -0.001851328 0.9979772
    5        atorvastatin   0.116301789 0.8707328
    6         glucosamine   1.059994442 0.1990808
    7        lansoprazole -16.116506142 0.9888694
    8 bendroflumethiazide  -1.027919542 0.3589906
    9       levothyroxine  -0.779117535 0.4958791



### PD infection models

#### Complex models


```r
%%R
pd_drugs_dt = subset(ad_pd, PD_diag == TRUE)

simvastatin = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    simvastatin,
    family = "binomial", data = pd_drugs_dt)
omeprazole = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    omeprazole,
    family = "binomial", data = pd_drugs_dt)
ramipril = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    ramipril,
    family = "binomial", data = pd_drugs_dt)
metformin = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    metformin,
    family = "binomial", data = pd_drugs_dt)
atorvastatin = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    atorvastatin,
    family = "binomial", data = pd_drugs_dt)
pramipexole = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    pramipexole,
    family = "binomial", data = pd_drugs_dt)
glucosamine = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    glucosamine,
    family = "binomial", data = pd_drugs_dt)
lansoprazole = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    lansoprazole,
    family = "binomial", data = pd_drugs_dt)
bendroflumethiazide = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    bendroflumethiazide,
    family = "binomial", data = pd_drugs_dt)
ropinirole = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    ropinirole,
    family = "binomial", data = pd_drugs_dt)
levothyroxine = glm(formula = result ~ n_cancers + townsend + diabetes + 
    whr + sex + age + ethnicity_sim + 
    log_crp +
    levothyroxine,
    family = "binomial", data = pd_drugs_dt)


PD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
                                  "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
                                  "ropinirole",
                                  "levothyroxine"),
                        OR = c(summary(simvastatin)$coefficients[10,1],
                                 summary(omeprazole)$coefficients[10,1],
                                 summary(ramipril)$coefficients[10,1],
                                 summary(metformin)$coefficients[10,1],
                                 summary(atorvastatin)$coefficients[10,1],
                                 summary(pramipexole)$coefficients[10,1],
                                 summary(glucosamine)$coefficients[10,1],
                                 summary(lansoprazole)$coefficients[10,1],
                                 summary(bendroflumethiazide)$coefficients[10,1],
                                 summary(ropinirole)$coefficients[10,1],
                                 summary(levothyroxine)$coefficients[10,1]),
                        p_val = c(summary(simvastatin)$coefficients[10,4],
                                 summary(omeprazole)$coefficients[10,4],
                                 summary(ramipril)$coefficients[10,4],
                                 summary(metformin)$coefficients[10,4],
                                 summary(atorvastatin)$coefficients[10,4],
                                 summary(pramipexole)$coefficients[10,4],
                                 summary(glucosamine)$coefficients[10,4],
                                 summary(lansoprazole)$coefficients[10,4],
                                 summary(bendroflumethiazide)$coefficients[10,4],
                                 summary(ropinirole)$coefficients[10,4],
                                 summary(levothyroxine)$coefficients[10,4]))
PD_drugs_df

```


                      drug           OR     p_val
    1          simvastatin   0.25023003 0.7005671
    2           omeprazole  -1.16952379 0.3544743
    3             ramipril  -0.98028857 0.3899831
    4            metformin   0.11757809 0.9999603
    5         atorvastatin -16.07450429 0.9939753
    6          pramipexole  -0.27469056 0.7583992
    7          glucosamine  -1.02818106 0.3594908
    8         lansoprazole  -0.08191310 0.9463307
    9  bendroflumethiazide   0.07602376 0.9351695
    10          ropinirole   0.88815282 0.2796343
    11       levothyroxine  -0.73294971 0.5415217



#### Simplified model


```r
%%R

simvastatin = glm(formula = result ~ whr + sex + age +
    simvastatin,
    family = "binomial", data = pd_drugs_dt)
omeprazole = glm(formula = covid_death ~ whr + sex + age +
    omeprazole,
    family = "binomial", data = pd_drugs_dt)
ramipril = glm(formula = result ~ whr + sex + age +
    ramipril,
    family = "binomial", data = pd_drugs_dt)
metformin = glm(formula = result ~ whr + sex + age +
    metformin,
    family = "binomial", data = pd_drugs_dt)
atorvastatin = glm(formula = result ~ whr + sex + age +
    atorvastatin,
    family = "binomial", data = pd_drugs_dt)
pramipexole = glm(formula = result ~ whr + sex + age +
    pramipexole,
    family = "binomial", data = pd_drugs_dt)
glucosamine = glm(formula = result ~ whr + sex + age +
    glucosamine,
    family = "binomial", data = pd_drugs_dt)
lansoprazole = glm(formula = result ~ whr + sex + age +
    lansoprazole,
    family = "binomial", data = pd_drugs_dt)
bendroflumethiazide = glm(formula = result ~ whr + sex + age +
    bendroflumethiazide,
    family = "binomial", data = pd_drugs_dt)
ropinirole = glm(formula = result ~ whr + sex + age +
    ropinirole,
    family = "binomial", data = pd_drugs_dt)
levothyroxine = glm(formula = result ~ whr + sex + age +
    levothyroxine,
    family = "binomial", data = pd_drugs_dt)


PD_drugs_df = data.frame(drug = c("simvastatin","omeprazole","ramipril",
                                  "metformin","atorvastatin",
                                  "pramipexole",
                                  "glucosamine","lansoprazole",
                                  "bendroflumethiazide",
                                  "ropinirole",
                                  "levothyroxine"),
                         OR = c(summary(simvastatin)$coefficients[5,1],
                                 summary(omeprazole)$coefficients[5,1],
                                 summary(ramipril)$coefficients[5,1],
                                 summary(metformin)$coefficients[5,1],
                                 summary(atorvastatin)$coefficients[5,1],
                                 summary(pramipexole)$coefficients[5,1],
                                 summary(glucosamine)$coefficients[5,1],
                                 summary(lansoprazole)$coefficients[5,1],
                                 summary(bendroflumethiazide)$coefficients[5,1],
                                 summary(ropinirole)$coefficients[5,1],
                                 summary(levothyroxine)$coefficients[5,1]),
                        p_val = c(summary(simvastatin)$coefficients[5,4],
                                 summary(omeprazole)$coefficients[5,4],
                                 summary(ramipril)$coefficients[5,4],
                                 summary(metformin)$coefficients[5,4],
                                 summary(atorvastatin)$coefficients[5,4],
                                 summary(pramipexole)$coefficients[5,4],
                                 summary(glucosamine)$coefficients[5,4],
                                 summary(lansoprazole)$coefficients[5,4],
                                 summary(bendroflumethiazide)$coefficients[5,4],
                                 summary(ropinirole)$coefficients[5,4],
                                 summary(levothyroxine)$coefficients[5,4]))
PD_drugs_df

```


                      drug           OR      p_val
    1          simvastatin   0.24650610 0.64874657
    2           omeprazole   1.74720240 0.09091277
    3             ramipril  -1.27956865 0.23458240
    4            metformin  -0.20326644 0.85681866
    5         atorvastatin -15.85869318 0.99028600
    6          pramipexole  -0.61906401 0.44903942
    7          glucosamine  -0.83502936 0.43646734
    8         lansoprazole  -0.90239215 0.39972759
    9  bendroflumethiazide   0.03655082 0.96506836
    10          ropinirole   0.45563021 0.52247216
    11       levothyroxine  -0.50047378 0.66130760




```r
%%R

ukb_covid_death_covid_red.b_odds = 
data.frame(cbind(exp(cbind(OR = coef(ukb_covid_death_covid_red.b), 
                           conf_int_low = summary(ukb_covid_death_covid_red.b)$coefficients[,1] - 
                                          summary(ukb_covid_death_covid_red.b)$coefficients[,2], 
                           conf_int_high = summary(ukb_covid_death_covid_red.b)$coefficients[,1] + 
                                          summary(ukb_covid_death_covid_red.b)$coefficients[,2])),
                 p_value = summary(ukb_covid_death_covid_red.b)$coefficients[,4]))

write.csv(ukb_covid_death_covid_red.b_odds,"data_output/ukb_covid_deathAndCOVID_odds_simplifiedModel.csv", row.names=TRUE)

ukb_covid_death_covid_red.b_odds$significance = "p-value > 0.05"
ukb_covid_death_covid_red.b_odds$significance[ukb_covid_death_covid_red.b_odds$p_value < 0.05] <- "p-value < 0.05"
ukb_covid_death_covid_red.b_odds$variables = row.names(ukb_covid_death_covid_red.b_odds)

library(ggplot2)
ggplot(ukb_covid_death_covid_red.b_odds, 
       aes(x=reorder(variables, OR), y=OR, color = significance)) + 
    geom_point(shape=21, size = 2) +
    geom_errorbar(aes(ymin=conf_int_low, ymax=conf_int_high),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
  theme_classic() + 
  geom_hline(yintercept = 1, linetype="dotted") +
  coord_flip()+ylab("Mortality odds ratios, death and COVID-19, simplified model") + 
  xlab("")
ggsave('fig_out/UKB_deathAndCOVID_ORplot_simplifiedModel.png', width = 6, height = 4)
```

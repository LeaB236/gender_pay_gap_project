majority_vote <- function(df) {
  # Calculate the table of frequencies for each row
  freq_table <- table(df)
  freq_table <- as.data.frame(freq_table)
  colnames(freq_table) <- c('label','freq')
  freq_table <- freq_table %>%
    arrange(desc(freq))
  majority <- freq_table$label[1]
  return(majority)
}

estimate.vector <- function(i, ls_ds_interaction, all_dummys){
  result <- (all_dummys[,ls_ds_interaction[[i]][1]]*all_dummys[,ls_ds_interaction[[i]][2]])
  return(result)
}

simple_corr_matrix_plot <- function(df ,label_size, size_coef, title_graph, colnames_df, method ) {
  df <- df %>% 
    select_if(is.numeric)
  #change colnames
  colnames(df)<- colnames_df
  #correlation matrix
  corr_matrix <- cor(df, method = method)
  corrplot(corr_matrix , type="lower",
           tl.cex= label_size,
           tl.offset = 0.5, tl.col = "black", method = "color",
           number.cex= 1, tl.srt = 45,
           addCoef.col = "black",
           diag = FALSE,
           title = title_graph, outline=FALSE, cl.pos="b",
           number.digits=2, mar=c(0,0,0,0))
}
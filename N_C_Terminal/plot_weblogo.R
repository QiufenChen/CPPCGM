# install.packages("installr")
# library(installr)
# updateR()
# install.packages("ggplot2")
# install.packages("devtools")
# devtools::install.github("omarwagih/ggseqlogo")


library(ggseqlogo)
library(ggplot2)


setwd("E:/CPPCGM/N_C_Terminal/")

# Create 'Visualization' folder if it doesn't exist
visualization_path <- "./Visualization"
if (!dir.exists(visualization_path)) {
  dir.create(visualization_path)
}

# Create a function that accepts a path and processes all txt files
plot_seqlogos <- function(path) {
  
  # Get all txt files in the directory
  files <- list.files(path, pattern = "\\.txt$", full.names = TRUE)
  
  # Loop through each file, read sequences, and plot sequence logos
  for (file in files) {
    # Read the sequences from the file
    sequences <- readLines(file)
    
    # Generate the sequence logo using ggseqlogo. 
    # The seq_type parameter specifies the sequence type. 
    # The default is "auto", which automatically detects the type. 
    # You can also choose from the following options: "aa" (amino acid), "dna" (DNA), or "rna" (RNA).
    seqlogo_plot <- ggseqlogo(sequences, facet = "wrap", seq_type="aa")
    
    # Set the output file path and name, save as a PDF in 'Visualization' folder
    output_file <- file.path(visualization_path, paste0(tools::file_path_sans_ext(basename(file)), "_logo.pdf"))
    
    # Save the plot as a PDF
    ggsave(output_file,  plot = seqlogo_plot, width = 7, height = 6, units = "in")
    # Print a message indicating the file has been saved
    cat("Saved plot for", basename(file), "as", output_file, "\n")
  }
}

# Call the function, passing the folder path
plot_seqlogos("./C_Terminal/")
plot_seqlogos("./N_Terminal/")

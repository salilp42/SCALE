                # Copy the correlation plot to the main figure
                ax_corr.clear()  # Clear existing content
                for child in corr_fig.axes[0].get_children():
                    if isinstance(child, plt.matplotlib.image.AxesImage):
                        ax_corr.imshow(child.get_array(), cmap=child.get_cmap())
                    elif isinstance(child, plt.matplotlib.text.Text):
                        # Correctly copy text properties
                        ax_corr.text(child.get_position()[0], 
                                   child.get_position()[1],
                                   child.get_text(),
                                   color=child.get_color(),
                                   fontsize=child.get_fontsize(),
                                   ha=child.get_horizontalalignment(),
                                   va=child.get_verticalalignment())
                
                ax_corr.set_title(corr_fig.axes[0].get_title()) 
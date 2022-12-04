import streamlit as st
import streamlit.components.v1 as components


def main():

    st.markdown("# Tableau Public Dashboard")

    tableau_public = '''<div class='tableauPlaceholder' id='viz1670088952747' style='position: relative' width=400 height=400><noscript><a href='#'><img alt='Sheet 7 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;He&#47;HeartDiseases_16682166124340&#47;Sheet7&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='HeartDiseases_16682166124340&#47;Sheet7' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;He&#47;HeartDiseases_16682166124340&#47;Sheet7&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div><script type='text/javascript'>var divElement = document.getElementById('viz1670088952747');var vizElement = divElement.getElementsByTagName('object')[0];vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';var scriptElement = document.createElement('script');scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';vizElement.parentNode.insertBefore(scriptElement, vizElement);</script>'''
    components.html(tableau_public)

if __name__ == '__main__':
    main()
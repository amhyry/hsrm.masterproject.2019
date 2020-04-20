'''
Created on 17 Jul 2019

@author: arnoldriemer
'''
#import Crawler
import wikipediaapi


def print_categorymembers(categorymembers, level=0, max_level=1):
        for c in categorymembers.values():
            print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
            if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                print_categorymembers(c.categorymembers, level=level + 1, max_level=max_level)

def print_sections(sections, level=0):
        for s in sections:
                print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text[0:40]))
                print_sections(s.sections, level + 1)



def print_categories(page):
    categories = page.categories
    for title in sorted(categories.keys()):
        print("%s: %s" % (title, categories[title]))

if __name__ == '__main__':
    #print("Hello")
    pass
    wiki_wiki = wikipediaapi.Wikipedia('de')


    page_py = wiki_wiki.page('Wirtschaftsinformatik')

    print("Page - Exists: %s" % page_py.exists())
    # Page - Exists: True

    print(page_py)
# Page - Exists: False

    print("Page - Summary: %s" % page_py.summary)

    print_sections(page_py.sections)




    #print("Categories")
    #print_categories(page_content)





    cat = wiki_wiki.page("Category:Physics")
    print("Category members: Category:Physics")
    print_categorymembers(cat.categorymembers)

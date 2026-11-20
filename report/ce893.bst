ENTRY
  { author
    title
    journal
    booktitle
    publisher
    year
    url
  }
  {}
  {}

FUNCTION {format.authors}
{ author empty$
    { "Anonymous" }
    { author #1 "{vv~}{ll}{, f.}" format.name$
      author num.names$ #1 >
        { " et~al" * }
        'skip$
      if$
    }
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
}

FUNCTION {finish.entry}
{ "." write$
  newline$
}

FUNCTION {article}
{ output.bibitem
  format.authors write$
  ". " write$
  title write$
  ". " write$
  journal empty$
    { skip$ }
    { "\emph{" write$ journal write$ "}" write$ }
  if$
  year empty$
    { skip$ }
    { ", " write$ year write$ }
  if$
  finish.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors write$
  ". " write$
  title write$
  ". In " write$
  booktitle empty$
    { "proceedings" write$ }
    { "\emph{" write$ booktitle write$ "}" write$ }
  if$
  year empty$
    { skip$ }
    { ", " write$ year write$ }
  if$
  finish.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors write$
  ". " write$
  title write$
  year empty$
    { skip$ }
    { ", " write$ year write$ }
  if$
  url empty$
    { skip$ }
    { ". \url{" write$ url write$ "}" write$ }
  if$
  finish.entry
}

FUNCTION {default.type} { misc }

READ

FUNCTION {begin.bib}
{ "\begin{thebibliography}{99}" write$ newline$ }

FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }

EXECUTE {begin.bib}
ITERATE {call.type$}
EXECUTE {end.bib}

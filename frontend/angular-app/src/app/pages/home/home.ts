import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Icon } from '../../shared/icon';
@Component({
  selector: 'app-home',
  imports: [RouterLink, Icon],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home {
  readonly tools = [
    { path: '/outils/recherche', icon: 'search', tag: 'RECHERCHER', title: 'Trouvez les bons textes', description: 'Explorez les documents juridiques à partir de mots-clés et de concepts.', action: 'Recherche lexicale' },
    { path: '/outils/classification', icon: 'layers', tag: 'ORGANISER', title: 'Donnez du sens au corpus', description: 'Découvrez comment classer un texte selon son domaine juridique.', action: 'Classification' },
    { path: '/outils/entites', icon: 'entities', tag: 'IDENTIFIER', title: 'Repérez l’essentiel', description: 'Retrouvez les personnes, les institutions et les références dans les textes.', action: 'Entités juridiques' },
    { path: '/outils/assistant', icon: 'spark', tag: 'COMPRENDRE', title: 'Éclairez vos questions', description: 'Découvrez une approche de la réponse juridique fondée sur ses sources.', action: 'Assistant juridique' },
  ];
}
